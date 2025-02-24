import os
import torch
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = "/Users/saeidmoghbel/datasets/qm9/qm9.db"
target_property = QM9.alpha
property_unit = {QM9.alpha: 'Bohr^3'}

qm9data = QM9(dataset_path,
              batch_size=16,
              num_train=110000,
              num_val=10000,
              num_test=None,
              transforms=[
                  trn.ASENeighborList(cutoff=5.),
                  trn.RemoveOffsets(QM9.alpha, remove_mean=True, remove_atomrefs=True),
                  trn.CastTo32()
              ],
              property_units=property_unit,
              num_workers=15,
              split_file=os.path.join(os.path.dirname(dataset_path), "split.npz")
)
qm9data.prepare_data()
qm9data.setup()

print(f"Number of total molecules: {len(qm9data.dataset)}")
print(f"Number of training samples: {len(qm9data.train_dataset)}")
print(f"Number of validation samples: {len(qm9data.val_dataset)}")
print(f"Number of test samples: {len(qm9data.test_dataset)}")

# Define SchNet model with Pytorch Lightning
class SchNetLightning(pl.LightningModule):
    def __init__(self, target, lr=1e-4):
        super().__init__()
        self.model = spk.representation.SchNet(
            n_atom_basis=128,
            n_filters=128,
            n_interactions=6,
            cutoff=5.0,
            n_gaussians=25,
        )
        self.output = spk.atomistic.Atomwise(n_in=128, property=target, derivative=None)
        self.lr = lr
        self.loss = torch.nn.MSELoss()
        self.target = target

    def forward(self, inputs):
        atoms_emb = self.model(inputs)
        return self.output(atoms_emb)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss(pred[self.target], batch[self.target])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss(pred[self.target], batch[self.target])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Initialize model and trainer
model = SchNetLightning(target=target_property).to(device)
logger = pl_loggers.TensorBoardLogger('logs')
trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
)

# Train the model
trainer.fit(model, qm9data.train_dataloader(), qm9data.val_dataloader())

# Function to extract molecular fingerprints
def get_fingerprint(batch, model):
    with torch.no_grad():
        atoms_emb = model.model(batch)
    return torch.mean(atoms_emb, dim=1)

# Example: Extract fingerprints from validation set
val_batch = next(iter(qm9data.val_dataloader()))
val_batch = {k: v.to(device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
fingerprints = get_fingerprint(val_batch, model)

# Analyze feature importance
readout_weights = model.output.outnet[0].weight.detach().cpu().numpy()
important_features = readout_weights.argsort()[:, ::-1]  # sort decreasing

print("Most important fingerprint indices for polarizability:", important_features[0][:10])