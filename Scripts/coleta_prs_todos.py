import requests
import csv
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
session = requests.Session()
if TOKEN:
    session.headers.update({"Authorization": f"token {TOKEN}"})



def github_request(url, params=None):
    while True:
        response = session.get(url, params=params)
        if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
            
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_time = reset_time - int(time.time()) + 1
            print(f" Rate limit atingido. Dormindo {sleep_time}s...")
            time.sleep(max(sleep_time, 1))
            continue  
        return response


def get_top_repositories(top_n=200):
    repos = []
    page = 1
    while len(repos) < top_n:
        url = "https://api.github.com/search/repositories"
        params = {"q": "stars:>1", "sort": "stars",
                  "order": "desc", "per_page": 100, "page": page}
        response = github_request(url, params=params)
        if response.status_code != 200:
            print("Erro ao buscar repositórios:", response.status_code)
            break
        data = response.json()
        repos.extend(data.get("items", []))
        page += 1
    return repos[:top_n]


def get_pull_requests(repo_full_name):
    """Coleta TODOS os PRs fechados (merged + closed sem merge)"""
    prs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo_full_name}/pulls"
        params = {"state": "closed", "per_page": 100, "page": page}
        response = github_request(url, params=params)
        if response.status_code != 200:
            print(f"Erro ao coletar PRs de {repo_full_name}: {response.status_code}")
            break

        data = response.json()
        if not data: 
            break

        prs.extend(data)
        page += 1

    return prs


def get_pr_files(repo_full_name, pr_number):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    response = github_request(url)
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
    comments_resp = github_request(url_comments)
    if comments_resp.status_code == 200:
        for c in comments_resp.json():
            if "user" in c and c["user"]:
                participants.add(c["user"].get("login", ""))

    url_reviews = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    reviews_resp = github_request(url_reviews)
    if reviews_resp.status_code == 200:
        for r in reviews_resp.json():
            if "user" in r and r["user"]:
                participants.add(r["user"].get("login", ""))
    return len(participants)


def get_comments_counts(repo_full_name, pr_number):
    url_issue = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}"
    resp = github_request(url_issue)
    if resp.status_code != 200:
        return 0, 0
    issue_data = resp.json()
    num_comments = issue_data.get("comments", 0)

    url_reviews = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    r = github_request(url_reviews)
    num_review_comments = len(r.json()) if r.status_code == 200 else 0

    return num_comments, num_review_comments


def process_pr(pr, repo_full_name):
    try:
        created = datetime.fromisoformat(pr["created_at"][:-1])
        closed = datetime.fromisoformat(
            pr["closed_at"][:-1]) if pr["closed_at"] else None
        if not closed:
            return None

        tempo_analise = (closed - created).total_seconds() / 3600
        if tempo_analise < 1:
            return None

        num_files, additions, deletions = get_pr_files(
            repo_full_name, pr["number"])
        num_participants = get_participants(repo_full_name, pr["number"])
        num_comments, num_review_comments = get_comments_counts(
            repo_full_name, pr["number"])

        state_final = "merged" if pr.get("merged_at") else "closed"

        return {
            "id": pr.get("id", 0),
            "repo": repo_full_name,
            "state": state_final,
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


def process_all_prs(repo_name, prs, writer):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_pr, pr, repo_name) for pr in prs]
        for future in as_completed(futures):
            dados = future.result()
            if dados:
                writer.writerow(dados)


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
            print(f" Coletando PRs do repositório: {repo_name}")
            prs = get_pull_requests(repo_name)
            process_all_prs(repo_name, prs, writer)

    print(" Coleta concluída! Dados salvos em dataset.csv")


if __name__ == "__main__":
    main()
