import os
import re
from turtle import position
import requests
import threading
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

def list_imagens_saved(base_path):
    arquivos = []
    caminhos = [os.path.join(base_path, nome) for nome in os.listdir(base_path)]
    for caminho in caminhos:
        arquivos.extend([nome for nome in os.listdir(caminho)])

    maior_por_animal = {}

    # Regex para extrair o animal e o número
    regex = re.compile(r'(\w+)_(\d+)\.jpg')

    for arquivo in arquivos:
        match = regex.match(arquivo)
        if match:
            animal, numero = match.groups()
            numero = int(numero)
            if animal not in maior_por_animal or numero > maior_por_animal[animal]:
                maior_por_animal[animal] = numero
    return maior_por_animal

def search_google_images(query, cx, key, start, num_images=10):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,  # Consulta
        "cx": cx,  # Substitua pelo ID correto
        "key": key,  # Substitua pela chave correta
        "searchType": "image",  # Tipo de pesquisa
        "num": num_images,  # Número de resultados
        "start": start
    }
    response = requests.get(search_url, params=params)
    results = response.json()
    #print(results)
    image_links = [item['link'] for item in results.get('items', [])]
    return image_links

def download_and_convert_image(url, file_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Carregar a imagem na memória
            img = Image.open(BytesIO(response.content))
            
            # Converter para JPG, se necessário
            if img.format != "JPEG":
                img = img.convert("RGB")
                file_path = file_path.replace('.jpg', '_converted.jpg')
            
            # Salvar a imagem no caminho especificado
            img.save(file_path, "JPEG")
        else:
            print(f"Falha ao baixar a imagem de {url}")
    except Exception as e:
        print(f"Erro ao baixar/converter a imagem de {url}: {e}")

def process_images(parametro, cx, key, base_path, position, num_images=10):
    # Diretório para salvar as imagens
    path = f'{base_path}/{parametro}'
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777, exist_ok=True)

    position += 1 

    image_links = []
    for start in range(position, position+num_images, 10):
        print(start, parametro)
        image_links.extend(search_google_images(parametro, cx, key, start=start, num_images=10))

    # Baixar as imagens usando threads
    download_threads = []
    for idx, link in enumerate(image_links):
        file_path = f'{path}/{parametro}_{idx + position}.jpg'
        thread = threading.Thread(target=download_and_convert_image, args=(link, file_path))
        download_threads.append(thread)
        thread.start()

    # Aguardar conclusão de todos os downloads
    for thread in download_threads:
        thread.join()

if __name__ == '__main__':
    load_dotenv(override=True)

    # Parâmetros de busca
    parametros = ['cat', 'dog', 'horse', 'fish', 'turtle']
    base_path = "db/images"
    cx = os.getenv('ENGINE_ID')
    key = os.getenv('API_KEY')

    maior_por_animal = list_imagens_saved(base_path)

    # Criar threads para cada parâmetro
    search_threads = []
    for parametro in parametros:        
        thread = threading.Thread(target=process_images, args=(parametro, cx, key, base_path, maior_por_animal.get(parametro, 1), 30))
        search_threads.append(thread)
        thread.start()

    # Aguardar conclusão de todas as buscas e downloads
    for thread in search_threads:
        thread.join()

    print("Todas as imagens foram processadas!")