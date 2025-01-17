import os
import shutil
import random

def distribute_files_randomly(source_folder, dest_folder_x, dest_folder_y):
    # Cria as pastas de destino, se não existirem
    os.makedirs(dest_folder_x, exist_ok=True)
    os.makedirs(dest_folder_y, exist_ok=True)

    # Lista todos os arquivos na pasta de origem
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Distribui os arquivos de forma aleatória
    for file in files:
        source_path = os.path.join(source_folder, file)
        if random.choice([True, False]):
            dest_path = os.path.join(dest_folder_x, file)
        else:
            dest_path = os.path.join(dest_folder_y, file)

        shutil.move(source_path, dest_path)
        print(f"Movido: {file} -> {dest_path}")

if __name__ == "__main__":
    source_folder = "images"  # Substitua pelo caminho da pasta de origem
    dest_folder_x = "train"  # Substitua pelo caminho da pasta X
    dest_folder_y = "test"  # Substitua pelo caminho da pasta Y
    
    #distribute_files_randomly(source_folder, dest_folder_x, dest_folder_y)
    print("Distribuição concluída!")