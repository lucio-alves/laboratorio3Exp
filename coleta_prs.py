import requests
import csv
from datetime import datetime
import time

TOKEN = "gittoken"  
HEADERS = {"Authorization": f"token {TOKEN}"}

def get_top_repositories(top_n=200):
    repos = []
    page = 1
    while len(repos) < top_n:
        url = "https://api.github.com/search/repositories"
        params = {"q": "stars:>1", "sort": "stars", "order": "desc", "per_page": 100, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print("Erro ao buscar repositórios:", response.status_code)
            break
        data = response.json()
        repos.extend(data.get("items", []))
        page += 1
        time.sleep(1)  
    return repos[:top_n]


def get_pull_requests(repo_full_name, max_prs=50):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls"
    params = {"state": "closed", "per_page": max_prs}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Erro ao coletar PRs de {repo_full_name}: {response.status_code}")
        return []
    return response.json()


def get_pr_files(repo_full_name, pr_number):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return 0, 0, 0
    files = response.json()
    num_files = len(files)
    additions = sum([f.get("additions", 0) for f in files])
    deletions = sum([f.get("deletions", 0) for f in files])
    return num_files, additions, deletions


def get_participants(repo_full_name, pr_number):
    participants = set()
    
    url_comments = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    comments_resp = requests.get(url_comments, headers=HEADERS)
    if comments_resp.status_code == 200:
        for c in comments_resp.json():
            if "user" in c and c["user"]:
                participants.add(c["user"].get("login", ""))

    url_reviews = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    reviews_resp = requests.get(url_reviews, headers=HEADERS)
    if reviews_resp.status_code == 200:
        for r in reviews_resp.json():
            if "user" in r and r["user"]:
                participants.add(r["user"].get("login", ""))
    return len(participants)


def get_comments_counts(repo_full_name, pr_number):
    url_issue = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}"
    resp = requests.get(url_issue, headers=HEADERS)
    if resp.status_code != 200:
        return 0, 0
    issue_data = resp.json()
    num_comments = issue_data.get("comments", 0)
    
    url_reviews = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    r = requests.get(url_reviews, headers=HEADERS)
    num_review_comments = len(r.json()) if r.status_code == 200 else 0
    
    return num_comments, num_review_comments

def process_pr(pr, repo_full_name):
    try:
        created = datetime.fromisoformat(pr["created_at"][:-1])
        closed = datetime.fromisoformat(pr["closed_at"][:-1]) if pr["closed_at"] else None
        if not closed:
            return None

        tempo_analise = (closed - created).total_seconds() / 3600
        if tempo_analise < 1:  
            return None

        num_files, additions, deletions = get_pr_files(repo_full_name, pr["number"])
 
        num_participants = get_participants(repo_full_name, pr["number"])

        num_comments, num_review_comments = get_comments_counts(repo_full_name, pr["number"])

        return {
            "id": pr.get("id", 0),
            "repo": repo_full_name,
            "state": pr.get("state", ""),
            "tempo_analise_horas": round(tempo_analise, 2),
            "descricao_len": len(pr["body"]) if pr.get("body") else 0,
            "num_comentarios": num_comments,
            "num_review_comments": num_review_comments,
            "autor": pr["user"].get("login", "") if pr.get("user") else "",
            "num_files": num_files,
            "additions": additions,
            "deletions": deletions,
            "num_participants": num_participants
        }
    except Exception as e:
        print(f"Erro ao processar PR: {e}")
        return None


def main():
    repos = get_top_repositories(200)

    with open("dataset.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id", "repo", "state", "tempo_analise_horas", "descricao_len",
            "num_comentarios", "num_review_comments", "autor",
            "num_files", "additions", "deletions", "num_participants"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repos:
            repo_name = repo["full_name"]
            print(f"Coletando PRs do repositório: {repo_name}")
            prs = get_pull_requests(repo_name, max_prs=50)
            for pr in prs:
                dados = process_pr(pr, repo_name)
                if dados:
                    writer.writerow(dados)
            time.sleep(1) 

    print("✅ Coleta concluída! Dados salvos em dataset.csv")

if __name__ == "__main__":
    main()
