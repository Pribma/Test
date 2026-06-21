from google.colab import drive
drive.mount('/content/drive')

!unzip -q data_souteze.zip -d rozbalena_data

# Tvoje cesta k datům bude začínat /content/ (to je hlavní složka Colabu)
train_ds = UniversalDataset("/content/rozbalena_data/training_set", is_test=False)

!unzip -q "/content/drive/MyDrive/MojeSlozka/data_souteze.zip" -d "/content/rozbalena_data"

train_ds = UniversalDataset("cesta/k/train", is_test=False)
Změníš ho prostě na:

Python
train_ds = UniversalDataset("/content/rozbalena_data/training_set", is_test=False)


train_ds = UniversalDataset("/content/rozbalena_data/moje_trenovaci_data.csv", is_test=False)

import pandas as pd
from torch.utils.data import Dataset

class UniversalDataset(Dataset):
    def __init__(self, file_path, is_test=False):
        self.is_test = is_test
        # Tady Python vezme tvou přesnou cestu k CSV a Pandas z ní udělá tabulku v paměti
        self.data = pd.read_csv(file_path)

    def __len__(self):
        # Vrátí počet řádků v té CSV tabulce
        return len(self.data)

    # ... (zbytek třídy, kde v __getitem__ rozebíráš jednotlivé řádky)
