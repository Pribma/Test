from google.colab import drive
drive.mount('/content/drive')

!unzip -q data_souteze.zip -d rozbalena_data

# Tvoje cesta k datům bude začínat /content/ (to je hlavní složka Colabu)
train_ds = UniversalDataset("/content/rozbalena_data/training_set", is_test=False)