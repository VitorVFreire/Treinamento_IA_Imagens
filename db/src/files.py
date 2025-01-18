import os
from PIL import Image, UnidentifiedImageError
import re, random
import filecmp
import os
import json
from rembg import remove
import io

class Files():
    def __init__(self, path:str, folders:list, name_file:str=None):
        self.path = path
        self.paths = []
        self.files = []
        self.max_number = 0
        self.names = []
        self.folders = folders
        if name_file is not None:
            self.name_file = name_file
            self.images_db = self.ler_ou_criar_arquivo()
            self.list_modifile = ['_image_black_white', '_rm_background', '_flip_image', '_rotation_images']

    def write_json_file(self):
        print('Carregando Arquivos...')
        self.load_names()
        for name in self.names:
            for folder in self.folders:
                path = f'{self.path}/{folder}/{name}'
                self.files.extend(self.load_files_by_folder(path))

        print('Verificando os Arquivos...')
        for item in self.images_db:
            if item.get('file') in self.files and not item.get('modifications') is None:
                self.files.remove(item.get('file'))

        print('Modificando os Arquivos...')
        for file in self.files:
            element = {
                'file': self.get_name(file),
                'path': file,
                'base_path': self.get_path(file),
                'modificações': []
            }

            for mod in self.list_modifile:
                mod_function = getattr(self, mod, None) 
                if callable(mod_function):
                    for name in mod_function(file, element['base_path'], element['file']):
                        mod_item = {
                            'mod_file': name,
                            'modification': mod
                        }
                        element['modifications'].append(mod_item)
            self.images_db.append(element)

        print('Salvando DB...')
        self.salvar_arquivo()

    def _image_black_white(self, file:str, base_path:str, file_name:str) -> list[str]:
        img=Image.open(file)
        blackAndWhite=img.convert("L")
        path = f'{base_path}/BLACK_WRITE_{file_name}'
        blackAndWhite.save(path)
        return [path]
    
    def _rm_background(self, file:str, base_path:str, file_name:str) -> list[str]:
        with open(file, "rb") as f:
            img_data = f.read()

        img_without_back_data = remove(img_data)
        img_without_back = Image.open(io.BytesIO(img_without_back_data)) 
        img_without_back = img_without_back.convert("RGB")
        path = f"{base_path}/BACKGROUND_{file_name}"
        img_without_back.save(path, "JPEG")

        return [path]
    
    def _flip_image(self, file:str, base_path:str, file_name:str) -> list[str]:
        img=Image.open(file)
        vertical_img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
        path = f'{base_path}/FLIP_BOTTOM_{file_name}'
        vertical_img.save(path)
        vertical_img_1 = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        path_1 = f'{base_path}/FLIP_RIGHT_{file_name}'
        vertical_img_1.save(path_1)
        return [path, path_1]

    def _rotation_images(self, file:str, base_path:str, file_name:str) -> list[str]:
        img=Image.open(file)
        rotations = [30, 60, 90, 120, 140, 160, 180]
        paths = []
        for rotation in rotations:
            path = f'{base_path}/ROTATION_{rotation}_{file_name}'
            rotated_img = img.rotate(rotation)
            rotated_img.save(path, "JPEG")
            paths.append(path)
        return paths
    
    def get_name(self, file) -> str:
        return re.split('/', file)[-1]
    
    def get_path(self, file) -> str:
        path_split = re.split('/', file)
        return '/'.join(path_split[:len(path_split)-1])

    def ler_ou_criar_arquivo(self):
        """Lê um arquivo JSON ou cria um arquivo vazio se ele não existir."""
        if not os.path.exists(self.name_file):
            with open(self.name_file, 'w') as arquivo:
                json.dump([], arquivo)  # Cria um JSON vazio (lista ou dicionário, conforme necessário)
            print(f"Arquivo '{self.name_file}' criado.")
        with open(self.name_file, 'r') as arquivo:
            try:
                dados = json.load(arquivo)
            except json.JSONDecodeError:
                dados = []  # Inicializa como lista vazia em caso de erro
                print("Arquivo estava vazio ou corrompido, inicializando com uma lista vazia.")
        return dados
    
    def salvar_arquivo(self):
        """Salva os dados no arquivo JSON."""
        with open(self.name_file, 'w') as arquivo:
            json.dump(self.images_db, arquivo, indent=4)
        print(f"Dados salvos em '{self.name_file}'.")

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
    
    def max_number_files(self, files):
        list_numbers = []
        regex = r'(\d+)\.'  # Expressão regular para capturar números antes do ponto.
        for file in files:
            match = re.search(regex, file)
            if match:
                file_number = int(match.group(1))
                list_numbers.append(file_number)
        return max(list_numbers, key=int)

    def count_files_per_folder(self) -> dict:
        try:
            data = {}
            self.load_names()
            files =[]
            for name in self.names:
                files.clear()
                for folder in self.folders:
                    path = f'{self.path}/{folder}/{name}'
                    files.extend(self.load_files_by_folder(path))
                data[name] = self.max_number_files(files)
            return data
        except OSError as e:
            return {}

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