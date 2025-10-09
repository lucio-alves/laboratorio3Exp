"""
Análise Visual Simples de Pull Requests do GitHub
================================================

Script focado na análise do dataset.csv com gráficos essenciais.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configurações de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Criar diretório de saída
output_dir = Path("graficos_analise")
output_dir.mkdir(exist_ok=True)

def carregar_dados():
    """Carrega e prepara os dados"""
    print("📊 Carregando dataset...")
    
    df = pd.read_csv("dataset.csv")
    print(f"✅ Dados carregados: {len(df):,} PRs de {df['repo'].nunique()} repositórios")
    
    # Criar coluna de tamanho total
    df['tamanho_total'] = df['additions'] + df['deletions']
    
    # Estatísticas básicas
    print(f"• Autores únicos: {df['autor'].nunique():,}")
    print(f"• Tempo médio análise: {df['tempo_analise_horas'].mean():.1f} horas")
    print(f"• Mediana arquivos/PR: {df['num_files'].median():.0f}")
    
    return df

def grafico_arquivos_vs_tempo(df):
    """Gráfico: Número de Arquivos × Tempo de Análise"""
    plt.figure(figsize=(12, 8))
    
    # Filtrar outliers extremos
    q99_files = df['num_files'].quantile(0.99)
    q99_time = df['tempo_analise_horas'].quantile(0.99)
    
    df_filtered = df[
        (df['num_files'] <= q99_files) & 
        (df['tempo_analise_horas'] <= q99_time)
    ]
    
    plt.scatter(df_filtered['num_files'], df_filtered['tempo_analise_horas'],
               alpha=0.6, s=50, c='#2E86AB', edgecolors='white', linewidth=0.5)
    
    # Linha de tendência
    z = np.polyfit(df_filtered['num_files'], df_filtered['tempo_analise_horas'], 1)
    p = np.poly1d(z)
    plt.plot(df_filtered['num_files'], p(df_filtered['num_files']), 
             color='#F18F01', linewidth=2, linestyle='--', 
             label=f'Tendência')
    
    plt.xlabel('Número de Arquivos Modificados', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
    plt.title(f'Relação: Arquivos × Tempo de Análise\n({len(df_filtered):,} PRs)', 
             fontsize=14, fontweight='bold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'arquivos_vs_tempo.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_tamanho_vs_tempo(df):
    """Gráfico: Tamanho Total × Tempo de Análise"""
    plt.figure(figsize=(12, 8))
    
    # Filtrar outliers
    q95_size = df['tamanho_total'].quantile(0.95)
    q95_time = df['tempo_analise_horas'].quantile(0.95)
    
    df_filtered = df[
        (df['tamanho_total'] <= q95_size) & 
        (df['tempo_analise_horas'] <= q95_time) &
        (df['tamanho_total'] > 0)
    ]
    
    # Scatter com cores por participantes
    scatter = plt.scatter(df_filtered['tamanho_total'], 
                         df_filtered['tempo_analise_horas'],
                         c=df_filtered['num_participants'],
                         cmap='viridis', alpha=0.7, s=60,
                         edgecolors='white', linewidth=0.5)
    
    plt.colorbar(scatter, label='Número de Participantes')
    
    plt.xlabel('Total de Linhas Modificadas (Adições + Deleções)', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
    plt.title(f'Relação: Tamanho do PR × Tempo de Análise\n({len(df_filtered):,} PRs)', 
             fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'tamanho_vs_tempo.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_top_autores(df):
    """Gráfico: Top 20 Autores"""
    plt.figure(figsize=(15, 10))
    
    author_counts = df['autor'].value_counts().head(20)
    
    bars = plt.barh(range(len(author_counts)), author_counts.values, 
                   color=sns.color_palette("viridis", len(author_counts)))
    
    plt.yticks(range(len(author_counts)), author_counts.index, fontsize=10)
    plt.xlabel('Número de Pull Requests', fontsize=12, fontweight='bold')
    plt.title(f'Top 20 Autores com Mais PRs\n(Total: {len(df):,} PRs)', 
             fontsize=14, fontweight='bold')
    
    # Valores nas barras
    for i, (bar, value) in enumerate(zip(bars, author_counts.values)):
        plt.text(value + 0.5, i, str(value), 
                va='center', ha='left', fontweight='bold', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'top_autores.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_distribuicao_repos(df):
    """Gráfico: Distribuição por Repositório"""
    plt.figure(figsize=(15, 8))
    
    repo_counts = df['repo'].value_counts().head(15)
    
    bars = plt.bar(range(len(repo_counts)), repo_counts.values,
                  color=sns.color_palette("Set2", len(repo_counts)))
    
    plt.xticks(range(len(repo_counts)), 
              [repo.split('/')[-1] for repo in repo_counts.index], 
              rotation=45, ha='right')
    plt.ylabel('Número de Pull Requests', fontsize=12, fontweight='bold')
    plt.title('Top 15 Repositórios por Número de PRs', fontsize=14, fontweight='bold')
    
    # Valores nas barras
    for bar, value in zip(bars, repo_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(value),
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'distribuicao_repos.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_estados_prs(df):
    """Gráfico: Estados dos PRs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Estados
    state_counts = df['state'].value_counts()
    colors = ['#C73E1D', '#7209B7', '#F18F01'][:len(state_counts)]
    
    ax1.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Distribuição de Estados dos PRs', fontweight='bold')
    
    # Tempo por estado
    avg_time = df.groupby('state')['tempo_analise_horas'].mean()
    bars = ax2.bar(avg_time.index, avg_time.values, color=colors[:len(avg_time)])
    
    ax2.set_ylabel('Tempo Médio de Análise (horas)', fontweight='bold')
    ax2.set_title('Tempo Médio por Estado', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Valores nas barras
    for bar, value in zip(bars, avg_time.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.0f}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    filename = output_dir / 'estados_prs.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_participacao(df):
    """Gráfico: Participação em PRs"""
    plt.figure(figsize=(12, 8))
    
    plt.hist(df['num_participants'], bins=range(0, df['num_participants'].max()+2),
            alpha=0.7, color='#2E86AB', edgecolor='white', linewidth=0.5)
    
    plt.xlabel('Número de Participantes por PR', fontsize=12, fontweight='bold')
    plt.ylabel('Frequência de PRs', fontsize=12, fontweight='bold')
    plt.title('Distribuição de Participação em PRs', fontsize=14, fontweight='bold')
    
    # Média
    mean_participants = df['num_participants'].mean()
    plt.axvline(mean_participants, color='#F18F01', linestyle='--', linewidth=2,
               label=f'Média: {mean_participants:.1f}')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'participacao.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def grafico_tempo_vs_comentarios(df):
    """Gráfico: Tempo vs Comentários"""
    plt.figure(figsize=(12, 8))
    
    # Filtrar para visualização
    q95_time = df['tempo_analise_horas'].quantile(0.95)
    q95_comments = df['num_comentarios'].quantile(0.95)
    
    df_filtered = df[
        (df['tempo_analise_horas'] <= q95_time) &
        (df['num_comentarios'] <= q95_comments)
    ]
    
    plt.scatter(df_filtered['num_comentarios'], 
               df_filtered['tempo_analise_horas'],
               alpha=0.6, c='#A23B72', s=50,
               edgecolors='white', linewidth=0.5)
    
    plt.xlabel('Número de Comentários', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
    plt.title(f'Relação: Comentários × Tempo de Análise\n({len(df_filtered):,} PRs)', 
             fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'tempo_vs_comentarios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Salvo: {filename}")

def main():
    """Função principal"""
    print("🎨 ANÁLISE VISUAL DE PULL REQUESTS - DATASET ORIGINAL")
    print("=" * 60)
    
    # Carregar dados
    df = carregar_dados()
    
    print("\n🎯 Gerando gráficos principais...")
    
    # Gráficos obrigatórios
    print("1/7 - Arquivos × Tempo")
    grafico_arquivos_vs_tempo(df)
    
    print("2/7 - Tamanho × Tempo") 
    grafico_tamanho_vs_tempo(df)
    
    print("3/7 - Top Autores")
    grafico_top_autores(df)
    
    # Gráficos adicionais
    print("4/7 - Distribuição Repos")
    grafico_distribuicao_repos(df)
    
    print("5/7 - Estados PRs")
    grafico_estados_prs(df)
    
    print("6/7 - Participação")
    grafico_participacao(df)
    
    print("7/7 - Tempo vs Comentários")
    grafico_tempo_vs_comentarios(df)
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA!")
    print(f"📁 Todos os gráficos foram salvos em: {output_dir}/")
    print(f"💡 Total de {len(df):,} PRs analisados de {df['repo'].nunique()} repositórios")

if __name__ == "__main__":
    main()