import tarfile

# Pfad zur .tar.xz-Datei
tar_xz_file_path = "/hpcwork/yv312705/NYU_data/knee_DICOMs_batch1.tar.xz"

# Öffnen der .tar.xz-Datei
with tarfile.open(tar_xz_file_path, 'r:xz') as tar:
    # Extrahieren aller Dateien im Archiv
    tar.extractall(path="/work/yv312705/NYU_data")

# Ausgabe der Extraktionserfolgmeldung
print("Die .tar.xz-Datei wurde erfolgreich extrahiert.")



