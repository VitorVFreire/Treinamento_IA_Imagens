from ast import Return
import os
from rembg import remove
from PIL import Image, UnidentifiedImageError
from filecmp import cmp as compare
from concurrent.futures import ThreadPoolExecutor
import cv2
import re, random
import shutil
import filecmp
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Files():
    def __init__(self, path):
        self.path = path
        self.paths = []
        self.files = []
        self.max_number = 0
        self.names = []
        self.folders = ['train', 'validation']
        self.files_by_folder = {}
        self.img_height = 150
        self.img_width = 150

    def get_only_files(self):
        self.files.extend([os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and 'png' in os.path.join(self.path, f)])

    def popular(self):
        self.load_names()
        copy_files = []

        for name in self.names:
            list_names = self.names.copy()  # Evita modificar a lista original
            list_names.remove(name)
            copy_files.clear()
            for folder in self.folders:
                path = f'{self.path}/{folder}/{name}'
                no_parametro_path = os.path.join(path, f'no_{name}')
                os.makedirs(no_parametro_path, mode=0o777, exist_ok=True)
                for other_name in list_names:
                    copy_files.extend(self.popular_folders(other_name, no_parametro_path, copy_files, name))

    def popular_folders(self, name: str, no_parametro_path: str, ignore_files: list, main_name: str) -> list:
        copy_files = []
        paths = []
        files = []
        for folder in self.folders:
            path = f'{self.path}/{folder}/{name}'
            paths, files = self.load_walks(path, main_name)
            copy_files.extend(self.copy_files(files, no_parametro_path, ignore_files))
        
        return copy_files

    def copy_files(self, files: list, dest_folder: str, ignore_files: list) -> list:
        # Remover arquivos a serem ignorados de forma segura
        files = [file for file in files if file not in ignore_files]

        tamanho_array = len(files)
        percent_copy = int(tamanho_array * 0.05)
        random.shuffle(files)
        
        copy_files = []

        for i in range(0, percent_copy):
            file = files[i]
            # Verifica se o arquivo de origem não é o mesmo que o arquivo de destino
            dest_file = os.path.join(dest_folder, os.path.basename(file))
            if file != dest_file:  # Verifica se o arquivo não é o mesmo
                copy_files.append(file)
                shutil.copy(file, dest_folder)
        
        return copy_files

    def load_walks(self, path: str, name: str) -> tuple[list[str], list[str]]:
        walk_paths = []
        walk_files = []
        for root, dirs, files in os.walk(path):
            # Adiciona o caminho de todas as pastas encontradas
            for d in dirs:
                if not 'no_' in d:
                    walk_paths.append(os.path.join(root, d))
            
            # Adiciona o caminho de todos os arquivos encontrados
            for f in files:
                if not 'no_' in f and not name in f:
                    walk_files.append(os.path.join(root, f))
        return walk_paths, walk_files

    def load_names(self):
        # Percorre as pastas 'train' e 'validation' para pegar os nomes das classes
        for folder in self.folders:
            folder_path = os.path.join(self.path, folder)
            # Verifica se o diretório existe
            if os.path.isdir(folder_path):
                # Lista as subpastas dentro de 'train' e 'validation' (nomes das classes)
                for class_name in os.listdir(folder_path):
                    class_path = os.path.join(folder_path, class_name)
                    if os.path.isdir(class_path) and not 'no_' in class_path and self.names.count(class_name) == 0:  # Verifica se é realmente um diretório
                        self.names.append(class_name)

    def load_files_and_paths(self):
        for root, dirs, files in os.walk(self.path):
            # Adiciona o caminho de todas as pastas encontradas
            for d in dirs:
                self.paths.append(os.path.join(root, d))
            
            # Adiciona o caminho de todos os arquivos encontrados
            for f in files:
                self.files.append(os.path.join(root, f))

    def __find_max_number_in_files(self):
        regex = r'(\d+)'  # Captura números em qualquer parte do nome do arquivo
        
        max_num = 0
        for file in self.files:
            match = re.findall(regex, os.path.basename(file))
            if match:
                numbers = map(int, match)  # Converte os números encontrados para inteiros
                max_num = max(max_num, max(numbers))  # Atualiza o maior número encontrado
        
        self.max_number = max_num

    def check_number_before_dot(self):
        self.__find_max_number_in_files()
        regex = r'(\d+)\.'  # Expressão regular para capturar números antes do ponto.

        for file in self.files:
            # Extrai o número antes do ponto no nome do arquivo.
            match = re.search(regex, os.path.basename(file))
            if match:
                file_number = int(match.group(1))
                if file_number > self.max_number and not ('FLIP' in file or 'ROTATED' in file):
                    self.new_files.append(file)
        
    def __is_corrupted(self, file_path):
        """Verifica se um arquivo de imagem está corrompido."""
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verifica se a imagem pode ser aberta sem erros
            return False
        except (UnidentifiedImageError, IOError):
            return True

    def __rm_background(self, path_file):
        """Remove o fundo de uma imagem usando rembg."""
        try:
            with Image.open(path_file) as img:
                img_without_back = remove(img)

                # Converte para RGB se a imagem estiver em RGBA
                if img_without_back.mode == "RGBA":
                    img_without_back = img_without_back.convert("RGB")

                img_without_back.save(path_file, "JPEG")
        except Exception as e:
            print(f"Erro ao remover fundo da imagem {path_file}: {e}")

    def load_files_by_folder(self, path: str) -> list:
        """Carrega arquivos organizados por pasta."""
        files_by_folder = []
        for root, _, files in os.walk(path):
            files_by_folder.extend([os.path.join(root, file) for file in files])
        return files_by_folder

    def rm_corrupted_files(self):
        """Remove arquivos corrompidos."""
        corrupted_files = [file for file in self.files if self.__is_corrupted(file)]
        for file in corrupted_files:
            os.remove(file)

    def rm_background_files(self):
        """Remove o fundo das imagens em múltiplas threads."""
        with ThreadPoolExecutor() as executor:
            executor.map(self.__rm_background, self.files)

    def flip_images(self):
        for path_file in self.files:
            try:
                # Carrega a imagem usando OpenCV
                img = cv2.imread(path_file)
                if img is None:
                    print(f"Erro ao carregar a imagem: {path_file}")
                    continue
                
                # Realiza o flip horizontal
                flipped_img = cv2.flip(img, 1)
                
                # Salva a imagem invertida
                flipped_path = os.path.join(os.path.dirname(path_file), f'FLIP_{os.path.basename(path_file)}')
                cv2.imwrite(flipped_path, flipped_img)
            except Exception as e:
                print(f"Erro ao processar a imagem {path_file}: {e}")
    
    def rotate_images(self):
        angles = [15, 30, 45, 60, 75, 90, 180]
        for path_file in self.files:
            try:
                # Carrega a imagem usando OpenCV
                img = cv2.imread(path_file)
                if img is None:
                    #print(f"Erro ao carregar a imagem: {path_file}")
                    continue
                
                # Obtém as dimensões da imagem
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                
                for angle in angles:
                    # Cria a matriz de rotação
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Aplica a rotação
                    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
                    
                    # Define o nome do arquivo de saída
                    rotated_path = os.path.join(
                        os.path.dirname(path_file),
                        f'ROTATED_{angle}_{os.path.basename(path_file)}'
                    )
                    
                    # Salva a imagem rotacionada
                    cv2.imwrite(rotated_path, rotated_img)
            except Exception as e:
                print(f"Erro ao processar a imagem {path_file}: {e}")

    def are_images_duplicate(self, image_path1, image_path2):
        # Verifica se ambos os arquivos existem antes de compará-los
        if not os.path.exists(image_path1):
            #print(f"Arquivo {image_path1} não encontrado!")
            return False
        if not os.path.exists(image_path2):
            #print(f"Arquivo {image_path2} não encontrado!")
            return False
        
        # Comparar os arquivos usando filecmp
        if filecmp.cmp(image_path1, image_path2, shallow=False):
            return True  # Arquivos idênticos
        else:
            return False  # Arquivos não idênticos

    # Método para remover duplicatas de maneira randômica
    def remove_duplicates(self, files):
        to_remove = []  # Lista para armazenar os arquivos que serão removidos

        # Iterar sobre os arquivos e comparar cada um com os outros
        for i, file1 in enumerate(files):
            if file1 in to_remove:  # Se já foi marcado para remoção, pular
                continue
            for j, file2 in enumerate(files[i+1:], i+1):
                if file2 in to_remove:  # Se já foi marcado para remoção, pular
                    continue
                # Comparar se os dois arquivos são duplicados
                if self.are_images_duplicate(file1, file2):
                    # Se for duplicata, marque um dos arquivos para remoção (de forma randômica)
                    if random.choice([True, False]):
                        to_remove.append(file2)
                    else:
                        to_remove.append(file1)
        
        print(f'{len(to_remove)} Arquivos para Remover...')

        # Deletar os arquivos duplicados
        for file in to_remove:
            try:
                os.remove(file)  # Deleta o arquivo
                #print(f"Arquivo removido: {file}")
            except OSError as e:
                print(f"Erro ao tentar remover o arquivo {file}: {e}")
        return len(to_remove)

    def run(self):
        '''print("Populando pastas...")
        self.load_files_and_paths()
        self.popular()

        print("Rotancionando Arquivos...")
        self.load_files_and_paths()
        self.rotate_images()

        print("Flipando Arquivos...")
        self.load_files_and_paths()
        self.flip_images()'''

        print("Removendo Arquivos Duplicados...")
        self.load_names()
        count_deleted = 0
        files = []
        for name in self.names:
            files.clear()
            for i in ['no_', '']:
                for folder in self.folders:
                    path = f'{self.path}/{folder}/{name}/{i}{name}'
                    files.extend(self.load_files_by_folder(path))
                count_deleted += self.remove_duplicates(files)

        print(f'{count_deleted} Arquivos Deletados!')

        print("Processamento concluído!")    