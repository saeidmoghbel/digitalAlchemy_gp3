import os
import ase
import schnetpack as spk
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList
import schnetpack.transform as trn
import ase.units
import torch
import torchmetrics
import pytorch_lightning as pl
from schnetpack.data import ASEAtomsData
from schnetpack.interfaces import AtomsConverter
from ase import Atoms


# ===== HPC-Compatible Paths =====
HOME_DIR = "/home/hpc/bccc/bccc135h/"
SCRATCH_DIR = "/scratch/hpc/bccc/bccc135h/"

data_path = os.path.join(SCRATCH_DIR, "qm9.db")
split_file_path = os.path.join(SCRATCH_DIR, "split_qm9.npz")
model_save_path = os.path.join(SCRATCH_DIR, "best_inference_model.pth")
log_path = os.path.join(SCRATCH_DIR, "logs")

# Ensure required directories exist
os.makedirs(log_path, exist_ok=True)

"""qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
  os.makedirs(qm9tut)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is using {device}")

# ===== QM9 Dataset Setup =====
if not os.path.exists(data_path):
    raise FileNotFoundError(f"QM9 dataset not found at {data_path}. Run `download_qm9.py` first!")


#!rm -rf qm9.db split_qm9.npz
qm9data = QM9(
    data_path,
    batch_size =10,
    num_train=110000,
    num_val=10000,
    split_file=split_file_path,
    transforms=[ASENeighborList(cutoff=5.)]
)
qm9data.prepare_data()
qm9data.setup()

print('Number of reference calculations:', len(qm9data.dataset))
print('Number of train data:', len(qm9data.train_dataset))
print('Number of test data:', len(qm9data.test_dataset))
print('Available properties:')

for p in qm9data.dataset.available_properties:
  print('-', p)

example = qm9data.dataset[0]
print('Properties:')

for k, v in example.items():
  print('-', k, ':', v.shape)

for batch in qm9data.val_dataloader():
  print(batch.keys())
  print('Isotropic Polarizability:', batch['isotropic_polarizability'])
  break

print('system index:', batch['_idx_m'])
print('Center atom index:', batch['_idx_i'])
print('Neighbor atom index:', batch['_idx_j'])


ase.units.a0 = ase.units.Bohr

qm9data = QM9(
    data_path,
    batch_size=100,
    num_train=1000,
    num_val=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(QM9.alpha, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units={QM9.alpha: 'Bohr'},
    num_workers=4,
    split_file=split_file_path,
    pin_memory=True,
    load_properties=[QM9.alpha],
)
qm9data.prepare_data()
qm9data.setup()

means, stddevs = qm9data.get_stats(
    QM9.alpha, divide_by_atoms=True, remove_atomref=False
)
print('Mean atomization energy / atoms:', means.item())
print('Std. dev. atomization energy / atom:', stddevs.item())

"""# Setting up the model"""

cutoff = 5.
n_atom_basis = 40

pairwise_distance = spk.atomistic.PairwiseDistances().to(device)  # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff).to(device)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=6,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
).to(device)
pred_alpha = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.alpha).to(device)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_alpha],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.alpha, add_mean=True, add_atomrefs=False)]
).to(device)

output_alpha = spk.task.ModelOutput(
    name=QM9.alpha,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE" : torchmetrics.MeanAbsoluteError(),
        "RMSE": torchmetrics.MeanSquaredError(squared=False)
    }
)

task = spk.task.AtomisticTask(
    model=nnpot.to(device),
    outputs=[output_alpha],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

"""## Training the model

The model is now ready for training. Since we already defined all necessary components, the only thing left to do is passing it to the pytorch Lightning Trainer together with the data module.
Additionally, we can provide callbacks that take care of logging, checkpointing etc.
"""

logger = pl.loggers.TensorBoardLogger(save_dir=log_path, name="logs")
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=model_save_path,
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    logger=False,
    default_root_dir=log_path,
    max_epochs=20,
)
trainer.fit(task, datamodule=qm9data)

"""# Inference

Having trained a model for QM9, we are going to use it to obtain some predictions. First, we need to load the model. The Trainer stores the best model in the model directory which can be loaded using PyTorch.
"""

import torch
import numpy as np
from ase import Atoms

best_model = torch.load(model_save_path, map_location=device)
best_model.to(device)

for batch in qm9data.test_dataloader():
  batch = {key: value.to(device) for key, value in batch.items()}

  with torch.no_grad():
    result = best_model(batch)
  print("Result dictionary:", result)
  break



def predict_polarizability(atoms_obj, model, device):
  """
  predicts the polarizability of a given molecular structure.

  Args:
      atoms_obj: ASE Atoms object representing a molecule.
      model: Trained SchNet model.
      device: The device where the model runs.

  Return:
      float: Predicted polarizability value.
  """
  # Ensure model is in evaluation mode
  model.eval()

  # Convert ASE Atoms  object to SchNetPack input format
  converter = AtomsConverter(
      neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32, device=device
  )
  inputs = converter(atoms_obj)
  inputs = {key: value.to(device) for key, value in inputs.items()} 
  
  #Run inference
  with torch.no_grad():
    result = model(inputs)

    # Debugging: Print available keys if an error occurs
    print("Model output keys:", result.keys())

    # Extract the predicted polarizability value
    predicted_polarizability = result[QM9.alpha].item()

    return predicted_polarizability


# Pick a molecule from the dataset
sample_data = qm9data.test_dataset[0]
atomic_numbers = sample_data["_atomic_numbers"].numpy()  # Convert tensors to NumPy
positions = sample_data["_positions"].numpy()   # Convert tensors to NumPy



converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)

numbers = np.array([6, 1, 1, 1, 1])
positions = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                      [0.002150416, -0.0060313176, 0.0019761204],
                      [1.0117308433, 1.4637511618, 0.0002765748],
                      [-0.540815069, 1.4475266138, -0.8766437152],
                      [-0.5238136345, 1.4379326443, 0.9063972942]])
atoms = Atoms(numbers=numbers, positions=positions)

inputs = converter(atoms)

print('Keys:', list(inputs.keys()))

# Predict Polarizability using model
with torch.no_grad():
    pred = best_model(inputs)

print('Prediction:', pred[QM9.alpha])
print(f"Predicted Polarizability: {pred[QM9.alpha].item():.4f}")

calculator = spk.interfaces.SpkCalculator(
    model_file=model_save_path,  # Path to model
    neighbor_list=trn.ASENeighborList(cutoff=5.),
    energy_key=QM9.alpha,  # Name of polarizability property in model
    energy_unit=1.0,  # Unit conversion factor
    device=device
)
atoms.set_calculator(calculator)
print('Prediction:', atoms.get_total_energy())

# Main execution
if __name__ == "__main__":
    trainer.fit(task, datamodule=qm9data)
    
    print("✅ Training Complete. Loading model for inference...")
    best_model = torch.load(model_save_path, map_location=device)
    best_model.to(device)
    best_model.eval()

    # Convert an ASE Atoms opbject
    sample_molecule = Atoms(numbers=atomic_numbers, positions=positions)

    # Call the function to predict polarizability
    predicted_value = predict_polarizability(sample_molecule, best_model, device)

    # Print the predicted value and the actual value
    actual_polarizability = sample_data[QM9.alpha].item()
    print(f"Predicted polarizability: {predicted_value}")
    print(f"Actual Polarizability: {actual_polarizability}")