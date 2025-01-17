import os
import random
import re
import threading
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO


class ImageProcessor():
    def __init__(self, cx, key, base_path, parametros, num_images=10):
        self.cx = cx
        self.key = key
        self.base_path = base_path
        self.parametros = parametros
        self.num_images = num_images
        self.paths = ['train', 'validation']

    def list_imagens_saved(self):
        """Lista a maior imagem salva para cada tipo de animal"""
        arquivos = []
        caminhos = [os.path.join(self.base_path, nome) for nome in os.listdir(self.base_path)]
        
        for caminho in caminhos:
            # Ignora arquivos .gitkeep e continua com diretórios
            if os.path.isdir(caminho):  # Garante que é um diretório
                arquivos.extend([nome for nome in os.listdir(caminho) if nome != '.gitkeep'])

        maior_por_animal = {}
        regex = re.compile(r'(\w+)_(\d+)\.jpg')

        for arquivo in arquivos:
            match = regex.match(arquivo)
            if match:
                animal, numero = match.groups()
                numero = int(numero)
                if animal not in maior_por_animal or numero > maior_por_animal[animal]:
                    maior_por_animal[animal] = numero
        return maior_por_animal

    def search_google_images(self, query, start):
        """Busca imagens no Google usando a API Custom Search"""
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": self.cx,
            "key": self.key,
            "searchType": "image",
            "num": 10,
            "start": start
        }
        response = requests.get(search_url, params=params)
        results = response.json()
        image_links = [item['link'] for item in results.get('items', [])]
        return image_links

    def download_and_convert_image(self, url, file_path):
        """Baixa e converte a imagem para o formato JPEG"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))

                if img.format != "JPEG":
                    img = img.convert("RGB")
                    file_path = file_path.replace('.jpg', '_converted.jpg')

                img.save(file_path, "JPEG")
            else:
                print(f"Falha ao baixar a imagem de {url}")
        except Exception as e:
            print(f"Erro ao baixar/converter a imagem de {url}: {e}")

    def process_images(self, parametro, start_position):
        """Processa o download das imagens para um parâmetro específico, balanceando entre train e validation"""
        image_links = []
        count_loop = (self.num_images / 10)

        # Proporção para distribuir imagens entre train e validation
        num_validation = int(self.num_images * 0.5)  # 50% para validation
        num_train = self.num_images - num_validation  # Restante para train

        # Lista aleatória de destinos balanceados
        destinations = ['validation'] * num_validation + ['train'] * num_train
        random.shuffle(destinations)

        for i, start in enumerate(range(start_position, start_position + self.num_images, 10)):
            image_links.extend(self.search_google_images(parametro, start=start))

        download_threads = []
        for idx, link in enumerate(image_links):
            if idx >= len(destinations):
                break

            # Define o destino balanceado
            destination = destinations[idx]
            base_path = f'{self.base_path}/{destination}/{parametro}/{parametro}'
            os.makedirs(base_path, mode=0o777, exist_ok=True)

            file_path = os.path.join(base_path, f"{parametro}_{idx + start_position}.jpg")
            thread = threading.Thread(target=self.download_and_convert_image, args=(link, file_path))
            download_threads.append(thread)
            thread.start()

        for thread in download_threads:
            thread.join()

    def run(self):
        """Inicia o processo de busca e download de imagens para todos os parâmetros"""
        maior_por_animal = self.list_imagens_saved()

        search_threads = []
        for parametro in self.parametros:
            start_position = maior_por_animal.get(parametro, 1)
            thread = threading.Thread(target=self.process_images, args=(parametro, start_position))
            search_threads.append(thread)
            thread.start()

        for thread in search_threads:
            thread.join()

        print("Todas as imagens foram processadas!")