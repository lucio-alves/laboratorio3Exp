"""
Análise Visual de Pull Requests do GitHub
=========================================

Script para análise e visualização de dados de PRs dos top 200 repositórios do GitHub.
Gera gráficos comparativos entre dataset.csv e dataset_completo.csv.

Autor: Análise de Dados GitHub PRs
Data: Outubro 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pathlib import Path
import os

# Configurações gerais
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurações de visualização
FIGSIZE_LARGE = (15, 10)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_SMALL = (10, 6)
DPI = 300

# Cores personalizadas
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'info': '#7209B7',
    'dark': '#2D3748'
}

class GitHubPRAnalyzer:
    """Classe para análise de dados de Pull Requests do GitHub"""
    
    def __init__(self, dataset_path, dataset_name):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.df = None
        self.output_dir = Path(f"graficos_{dataset_name.lower().replace('.csv', '')}")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Carrega e prepara os dados"""
        print(f"📊 Carregando dataset: {self.dataset_name}")
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            
            # Detectar e padronizar estrutura do dataset
            if 'repo_full_name' in self.df.columns:
                # Dataset completo - mapear colunas
                self.df = self._normalize_complete_dataset()
            else:
                # Dataset original - verificar se tem as colunas necessárias
                self.df = self._normalize_original_dataset()
            
            print(f"✅ Dados carregados: {len(self.df):,} PRs de {self.df['repo'].nunique()} repositórios")
            
            # Criar coluna de tamanho total (linhas modificadas)
            self.df['tamanho_total'] = self.df['additions'] + self.df['deletions']
            
            # Limpeza básica
            required_cols = ['tempo_analise_horas', 'num_files']
            self.df = self.df.dropna(subset=required_cols)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def _normalize_complete_dataset(self):
        """Normaliza o dataset completo para o formato padrão"""
        df_norm = self.df.copy()
        
        # Mapear colunas do dataset completo para o formato padrão
        column_mapping = {
            'repo_full_name': 'repo',
            'author': 'autor',
            'review_duration_hours': 'tempo_analise_horas',
            'changed_files': 'num_files',
            'description_length': 'descricao_len',
            'comments_count': 'num_comentarios',
            'participants_count': 'num_participants'
        }
        
        # Renomear colunas existentes
        for old_col, new_col in column_mapping.items():
            if old_col in df_norm.columns:
                df_norm[new_col] = df_norm[old_col]
        
        # Criar colunas ausentes com valores padrão
        if 'num_review_comments' not in df_norm.columns:
            df_norm['num_review_comments'] = 0  # Valor padrão
        
        if 'state' not in df_norm.columns:
            # Inferir estado baseado na coluna 'merged'
            df_norm['state'] = df_norm['merged'].apply(lambda x: 'merged' if x else 'closed')
        
        return df_norm
    
    def _normalize_original_dataset(self):
        """Normaliza o dataset original (verifica se está no formato correto)"""
        return self.df
    
    def get_summary_stats(self):
        """Retorna estatísticas resumidas do dataset"""
        if self.df is None:
            return None
            
        stats = {
            'total_prs': len(self.df),
            'repositorios': self.df['repo'].nunique(),
            'autores_unicos': self.df['autor'].nunique(),
            'tempo_medio_analise': self.df['tempo_analise_horas'].mean(),
            'mediana_arquivos': self.df['num_files'].median(),
            'total_linhas_codigo': self.df['tamanho_total'].sum()
        }
        
        return stats
    
    def print_summary(self):
        """Imprime resumo do dataset"""
        stats = self.get_summary_stats()
        if stats:
            print(f"\n📈 RESUMO - {self.dataset_name}")
            print("=" * 50)
            print(f"• Total de PRs: {stats['total_prs']:,}")
            print(f"• Repositórios únicos: {stats['repositorios']}")
            print(f"• Autores únicos: {stats['autores_unicos']:,}")
            print(f"• Tempo médio análise: {stats['tempo_medio_analise']:.2f} horas")
            print(f"• Mediana arquivos/PR: {stats['mediana_arquivos']:.0f}")
            print(f"• Total linhas modificadas: {stats['total_linhas_codigo']:,}")
    
    def plot_files_vs_time_analysis(self):
        """Gráfico: Número de Arquivos × Tempo de Análise"""
        plt.figure(figsize=FIGSIZE_MEDIUM)
        
        # Remover outliers extremos para melhor visualização
        q99_files = self.df['num_files'].quantile(0.99)
        q99_time = self.df['tempo_analise_horas'].quantile(0.99)
        
        df_filtered = self.df[
            (self.df['num_files'] <= q99_files) & 
            (self.df['tempo_analise_horas'] <= q99_time)
        ]
        
        # Scatter plot com transparência e cores por densidade
        plt.scatter(df_filtered['num_files'], 
                   df_filtered['tempo_analise_horas'],
                   alpha=0.6, 
                   s=50,
                   c=COLORS['primary'],
                   edgecolors='white',
                   linewidth=0.5)
        
        # Linha de tendência
        z = np.polyfit(df_filtered['num_files'], df_filtered['tempo_analise_horas'], 1)
        p = np.poly1d(z)
        plt.plot(df_filtered['num_files'], p(df_filtered['num_files']), 
                 color=COLORS['accent'], linewidth=2, linestyle='--', 
                 label=f'Tendência (R² = {np.corrcoef(df_filtered["num_files"], df_filtered["tempo_analise_horas"])[0,1]**2:.3f})')
        
        plt.xlabel('Número de Arquivos Modificados', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
        plt.title(f'Relação: Arquivos × Tempo de Análise\n{self.dataset_name} (n={len(df_filtered):,})', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'arquivos_vs_tempo_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_size_vs_time_analysis(self):
        """Gráfico: Tamanho Total (Linhas) × Tempo de Análise"""
        plt.figure(figsize=FIGSIZE_MEDIUM)
        
        # Filtrar outliers extremos
        q95_size = self.df['tamanho_total'].quantile(0.95)
        q95_time = self.df['tempo_analise_horas'].quantile(0.95)
        
        df_filtered = self.df[
            (self.df['tamanho_total'] <= q95_size) & 
            (self.df['tempo_analise_horas'] <= q95_time) &
            (self.df['tamanho_total'] > 0)
        ]
        
        # Scatter plot com gradiente de cores
        scatter = plt.scatter(df_filtered['tamanho_total'], 
                            df_filtered['tempo_analise_horas'],
                            c=df_filtered['num_participants'],
                            cmap='viridis',
                            alpha=0.7,
                            s=60,
                            edgecolors='white',
                            linewidth=0.5)
        
        # Barra de cores
        cbar = plt.colorbar(scatter)
        cbar.set_label('Número de Participantes', fontsize=10)
        
        # Linha de tendência logarítmica se necessário
        if df_filtered['tamanho_total'].max() > 1000:
            log_x = np.log1p(df_filtered['tamanho_total'])
            z = np.polyfit(log_x, df_filtered['tempo_analise_horas'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_filtered['tamanho_total'].min(), df_filtered['tamanho_total'].max(), 100)
            y_trend = p(np.log1p(x_trend))
            plt.plot(x_trend, y_trend, color=COLORS['secondary'], linewidth=2, linestyle='--', 
                    label='Tendência (log)')
        else:
            z = np.polyfit(df_filtered['tamanho_total'], df_filtered['tempo_analise_horas'], 1)
            p = np.poly1d(z)
            plt.plot(df_filtered['tamanho_total'], p(df_filtered['tamanho_total']), 
                    color=COLORS['secondary'], linewidth=2, linestyle='--', 
                    label='Tendência linear')
        
        plt.xlabel('Total de Linhas Modificadas (Adições + Deleções)', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
        plt.title(f'Relação: Tamanho do PR × Tempo de Análise\n{self.dataset_name} (n={len(df_filtered):,})', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'tamanho_vs_tempo_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_top_authors(self, top_n=20):
        """Gráfico: Top N Autores com Mais PRs"""
        plt.figure(figsize=FIGSIZE_LARGE)
        
        # Contar PRs por autor
        author_counts = self.df['autor'].value_counts().head(top_n)
        
        # Criar gráfico horizontal
        bars = plt.barh(range(len(author_counts)), author_counts.values, 
                       color=sns.color_palette("viridis", len(author_counts)))
        
        # Personalizar
        plt.yticks(range(len(author_counts)), author_counts.index, fontsize=10)
        plt.xlabel('Número de Pull Requests', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Autores com Mais PRs\n{self.dataset_name} (Total: {len(self.df):,} PRs)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, author_counts.values)):
            plt.text(value + 0.5, i, str(value), 
                    va='center', ha='left', fontweight='bold', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'top_autores_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_repository_distribution(self):
        """Gráfico: Distribuição de PRs por Repositório"""
        plt.figure(figsize=FIGSIZE_LARGE)
        
        repo_counts = self.df['repo'].value_counts().head(15)
        
        # Gráfico de barras vertical com cores diferentes
        bars = plt.bar(range(len(repo_counts)), repo_counts.values,
                      color=sns.color_palette("Set2", len(repo_counts)))
        
        plt.xticks(range(len(repo_counts)), 
                  [repo.split('/')[-1] for repo in repo_counts.index], 
                  rotation=45, ha='right')
        plt.ylabel('Número de Pull Requests', fontsize=12, fontweight='bold')
        plt.title(f'Top 15 Repositórios por Número de PRs\n{self.dataset_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, repo_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(value),
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'distribuicao_repos_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_pr_state_analysis(self):
        """Gráfico: Análise de Estados dos PRs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)
        
        # Subplot 1: Distribuição de estados
        state_counts = self.df['state'].value_counts()
        colors = [COLORS['success'], COLORS['info'], COLORS['accent']][:len(state_counts)]
        
        wedges, texts, autotexts = ax1.pie(state_counts.values, 
                                          labels=state_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          explode=[0.05] * len(state_counts))
        
        ax1.set_title(f'Distribuição de Estados dos PRs\n{self.dataset_name}', 
                     fontweight='bold', pad=20)
        
        # Subplot 2: Tempo médio por estado
        avg_time_by_state = self.df.groupby('state')['tempo_analise_horas'].mean()
        
        bars = ax2.bar(avg_time_by_state.index, avg_time_by_state.values,
                      color=colors[:len(avg_time_by_state)])
        
        ax2.set_ylabel('Tempo Médio de Análise (horas)', fontweight='bold')
        ax2.set_title('Tempo Médio por Estado', fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, avg_time_by_state.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'analise_estados_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_participation_analysis(self):
        """Gráfico: Análise de Participação em PRs"""
        plt.figure(figsize=FIGSIZE_MEDIUM)
        
        # Histograma de participação
        plt.hist(self.df['num_participants'], bins=range(0, self.df['num_participants'].max()+2),
                alpha=0.7, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
        
        plt.xlabel('Número de Participantes por PR', fontsize=12, fontweight='bold')
        plt.ylabel('Frequência de PRs', fontsize=12, fontweight='bold')
        plt.title(f'Distribuição de Participação em PRs\n{self.dataset_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar estatísticas
        mean_participants = self.df['num_participants'].mean()
        plt.axvline(mean_participants, color=COLORS['accent'], linestyle='--', linewidth=2,
                   label=f'Média: {mean_participants:.1f}')
        
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'analise_participacao_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def plot_comments_review_correlation(self):
        """Gráfico: Correlação entre Comentários e Review Comments"""
        plt.figure(figsize=FIGSIZE_MEDIUM)
        
        # Filtrar dados para melhor visualização
        max_comments = self.df['num_comentarios'].quantile(0.95)
        max_reviews = self.df['num_review_comments'].quantile(0.95)
        
        df_filtered = self.df[
            (self.df['num_comentarios'] <= max_comments) &
            (self.df['num_review_comments'] <= max_reviews)
        ]
        
        # Scatter plot com densidade
        plt.scatter(df_filtered['num_comentarios'], 
                   df_filtered['num_review_comments'],
                   alpha=0.6,
                   c=df_filtered['tempo_analise_horas'],
                   cmap='plasma',
                   s=50,
                   edgecolors='white',
                   linewidth=0.5)
        
        # Barra de cores
        cbar = plt.colorbar()
        cbar.set_label('Tempo de Análise (horas)', fontsize=10)
        
        plt.xlabel('Número de Comentários', fontsize=12, fontweight='bold')
        plt.ylabel('Número de Review Comments', fontsize=12, fontweight='bold')
        plt.title(f'Relação: Comentários × Review Comments\n{self.dataset_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar
        filename = self.output_dir / f'comentarios_reviews_{self.dataset_name.replace(".csv", "")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def generate_all_plots(self):
        """Gera todos os gráficos para o dataset"""
        print(f"\n🎨 Gerando gráficos para {self.dataset_name}")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        self.print_summary()
        
        # Gráficos obrigatórios
        print(f"\n📊 Gráfico 1/7: Arquivos × Tempo...")
        self.plot_files_vs_time_analysis()
        
        print(f"📊 Gráfico 2/7: Tamanho × Tempo...")
        self.plot_size_vs_time_analysis()
        
        print(f"📊 Gráfico 3/7: Top Autores...")
        self.plot_top_authors()
        
        # Gráficos adicionais
        print(f"📊 Gráfico 4/7: Distribuição Repositórios...")
        self.plot_repository_distribution()
        
        print(f"📊 Gráfico 5/7: Estados dos PRs...")
        self.plot_pr_state_analysis()
        
        print(f"📊 Gráfico 6/7: Análise Participação...")
        self.plot_participation_analysis()
        
        print(f"📊 Gráfico 7/7: Comentários × Reviews...")
        self.plot_comments_review_correlation()
        
        print(f"\n✅ Todos os gráficos foram gerados e salvos em: {self.output_dir}")
        return True

def create_comparison_summary(analyzer1, analyzer2):
    """Cria um resumo comparativo entre os dois datasets"""
    print("\n📋 COMPARAÇÃO ENTRE DATASETS")
    print("=" * 60)
    
    stats1 = analyzer1.get_summary_stats()
    stats2 = analyzer2.get_summary_stats()
    
    if stats1 and stats2:
        print(f"{'Métrica':<25} {'Dataset 1':<15} {'Dataset 2':<15} {'Diferença':<15}")
        print("-" * 70)
        
        metrics = [
            ('PRs Total', 'total_prs', ''),
            ('Repositórios', 'repositorios', ''),
            ('Autores Únicos', 'autores_unicos', ''),
            ('Tempo Médio (h)', 'tempo_medio_analise', 'h'),
            ('Mediana Arquivos', 'mediana_arquivos', ''),
            ('Total Linhas', 'total_linhas_codigo', '')
        ]
        
        for label, key, unit in metrics:
            val1 = stats1[key]
            val2 = stats2[key]
            
            if isinstance(val1, float):
                diff = f"{((val2/val1 - 1) * 100):+.1f}%" if val1 > 0 else "N/A"
                val1_str = f"{val1:.1f}{unit}"
                val2_str = f"{val2:.1f}{unit}"
            else:
                diff = f"{((val2/val1 - 1) * 100):+.1f}%" if val1 > 0 else "N/A"
                val1_str = f"{val1:,}{unit}"
                val2_str = f"{val2:,}{unit}"
            
            print(f"{label:<25} {val1_str:<15} {val2_str:<15} {diff:<15}")

def main():
    """Função principal"""
    print("🚀 ANÁLISE VISUAL DE PULL REQUESTS DO GITHUB")
    print("=" * 60)
    print("Analisando dados dos top 200 repositórios mais populares")
    print("Gerando visualizações comparativas dos datasets\n")
    
    # Caminhos dos datasets
    datasets = [
        ("dataset.csv", "Dataset Original"),
        ("dataset_completo.csv", "Dataset Completo")
    ]
    
    analyzers = []
    
    # Processar cada dataset
    for filename, display_name in datasets:
        if os.path.exists(filename):
            print(f"🔍 Processando: {filename}")
            analyzer = GitHubPRAnalyzer(filename, filename)
            
            try:
                if analyzer.generate_all_plots():
                    analyzers.append(analyzer)
                else:
                    print(f"❌ Falha ao processar {filename}")
            except Exception as e:
                print(f"❌ Erro ao processar {filename}: {e}")
                print(f"⚠️  Continuando com próximo dataset...")
        else:
            print(f"⚠️  Arquivo não encontrado: {filename}")
    
    # Criar comparação se ambos datasets foram processados
    if len(analyzers) == 2:
        create_comparison_summary(analyzers[0], analyzers[1])
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA!")
    print(f"📁 Gráficos salvos nos diretórios:")
    for analyzer in analyzers:
        print(f"   • {analyzer.output_dir}/")
    
    print(f"\n💡 Dica: Abra os arquivos .png para visualizar os gráficos gerados!")

if __name__ == "__main__":
    main()