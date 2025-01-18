import os
from PIL import Image, UnidentifiedImageError
import re, random
import filecmp

class Files():
    def __init__(self, path:str, folders:list):
        self.path = path
        self.paths = []
        self.files = []
        self.max_number = 0
        self.names = []
        self.folders = folders

    def get_only_files(self):
        self.files.extend([os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and not '.git' in os.path.join(self.path, f)])

    def load_names(self):
        # Percorre as pastas 'train' e 'validation' para pegar os nomes das classes
        for folder in self.folders:
            folder_path = os.path.join(self.path, folder)
            # Verifica se o diretório existe
            if os.path.isdir(folder_path):
                # Lista as subpastas dentro de 'train' e 'validation' (nomes das classes)
                for class_name in os.listdir(folder_path):
                    class_path = os.path.join(folder_path, class_name)
                    if os.path.isdir(class_path) and self.names.count(class_name) == 0:  # Verifica se é realmente um diretório
                        self.names.append(class_name)
        self.names.sort()

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

    def load_files_by_folder(self, path: str) -> list:
        """Carrega arquivos organizados por pasta."""
        files_by_folder = []
        for root, _, files in os.walk(path):
            files_by_folder.extend([os.path.join(root, file) for file in files])
        return files_by_folder

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

        # Deletar os arquivos duplicados
        for file in to_remove:
            try:
                os.remove(file)  # Deleta o arquivo
                #print(f"Arquivo removido: {file}")
            except OSError as e:
                print(f"Erro ao tentar remover o arquivo {file}: {e}")
        return len(to_remove)

    def run(self):
        print("Removendo Arquivos Corrompidos...")
        self.load_files_and_paths()
        for file in self.files:
            if self.__is_corrupted(file):
                os.remove(file)
        
        self.files.clear()

        print("Removendo Arquivos Duplicados...")
        self.load_names()
        count_deleted = 0
        files = []
        for name in self.names:
            files.clear()
            for folder in self.folders:
                path = f'{self.path}/{folder}/{name}'
                files.extend(self.load_files_by_folder(path))
            count_deleted += self.remove_duplicates(files)

        print(f'{count_deleted} Arquivos Deletados!')

        print("Processamento concluído!")    