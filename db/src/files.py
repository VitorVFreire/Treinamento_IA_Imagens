import os
from rembg import remove
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from filecmp import cmp as compare
from itertools import combinations
from networkx import Graph, connected_components
from concurrent.futures import ThreadPoolExecutor
import cv2
import re

class Files():
    def __init__(self, path):
        self.path = path
        self.paths = []
        self.files = []
        self.new_files = []
        self.max_number = 0
        self.names = []

    def load_names(self):
        self.paths = [path for path in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, path))]
        for path in self.paths:
            self.names.append(re.split("/", path, 1)[0])

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
        
    def __get_paths(self):
        """Recupera os diretórios dentro do diretório principal."""
        self.paths = [path for path in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, path))]

    def get_only_files(self):
        self.files.extend([os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and 'png' in os.path.join(self.path, f)])

    def __get_files(self):
        """Recupera todos os arquivos dentro dos diretórios listados."""
        for path in self.paths:
            full_path = os.path.join(self.path, path)
            self.files.extend([os.path.join(full_path, f) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])

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

    def __duplicados(self, files):
        """Encontra arquivos duplicados e retorna seus caminhos."""
        dups = [(f1, f2) for f1, f2 in combinations(files, 2) if compare(f1, f2)]

        # Constroi um grafo a partir dos pares de arquivos duplicados
        grafo = Graph()
        grafo.add_edges_from(dups)

        # Recupera os componentes conectados (arquivos duplicados)
        componentes = list(connected_components(grafo))

        return [list(componente) for componente in componentes]

    def delete_files(self):
        duplicates = []
        
        for path in self.paths:
            files_in_path = [str(p) for p in Path(f'{self.path}/{path}').resolve().glob('*.jpg')]
            duplicates.extend(self.__duplicados(files_in_path))

        for duplicate_group in duplicates:
            for file in duplicate_group[1:]:  # Mantém apenas o primeiro arquivo
                os.remove(file)

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
        for path_file in self.new_files:
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
        angles = [15, 30, 45, 60]
        for path_file in self.new_files:
            try:
                # Carrega a imagem usando OpenCV
                img = cv2.imread(path_file)
                if img is None:
                    print(f"Erro ao carregar a imagem: {path_file}")
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

    def run(self):
        self.__get_paths()
        self.__get_files()

        print("Verificando arquivos corrompidos...")
        self.rm_corrupted_files()

        print("Removendo arquivos duplicados...")
        self.delete_files()

        self.files.clear()
        self.__get_files()

        print("Flipando arquivos...")
        self.flip_images()

        self.files.clear()
        self.__get_files()
        
        print("Rotacionando arquivos...")
        self.rotate_images()

        print("Removendo arquivos duplicados...")
        self.files.clear()
        self.__get_files()
        self.delete_files()

        print("Processamento concluído!")