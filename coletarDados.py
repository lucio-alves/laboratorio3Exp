import requests
import csv
import time
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}


def get_top_repositories(top_n=200):
    repos = []
    page = 1
    while len(repos) < top_n:
        url = "https://api.github.com/search/repositories"
        params = {
            "q": "stars:>1",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(" Erro ao buscar repositórios:",
                  response.status_code, response.text)
            break

        data = response.json()
        repos.extend(data.get("items", []))
        print(f" Página {page} coletada, total até agora: {len(repos)}")
        page += 1
        time.sleep(1)  # pausa para evitar limite da API

    return repos[:top_n]


def salvar_csv(repos, filename="top200_repositorios.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["full_name", "html_url",
                      "stargazers_count", "language", "description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repos:
            writer.writerow({
                "full_name": repo["full_name"],
                "html_url": repo["html_url"],
                "stargazers_count": repo["stargazers_count"],
                "language": repo["language"],
                "description": repo["description"]
            })
    print(f" Arquivo '{filename}' criado com sucesso!")


def main():
    repos = get_top_repositories(200)
    salvar_csv(repos)


if __name__ == "__main__":
    main()
