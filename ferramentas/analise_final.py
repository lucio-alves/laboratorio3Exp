"""
Análise Completa dos Datasets de Pull Requests do GitHub
========================================================

Script final para análise dos datasets dataset.csv e dataset_completo.csv
com tratamento robusto de erros e gráficos comparativos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DatasetAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.stats = {}
        
    def load_data(self):
        """Carrega dados com tratamento robusto de erros"""
        try:
            print(f"📊 Carregando {self.filename}...")
            
            # Tentar diferentes encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(self.filename, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                print(f"❌ Não foi possível carregar {self.filename}")
                return False
            
            # Normalizar colunas baseado no tipo de dataset
            self._normalize_columns()
            
            # Calcular estatísticas
            self._calculate_stats()
            
            print(f"✅ {self.filename}: {len(self.df):,} PRs de {self.df['repo'].nunique()} repositórios")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar {self.filename}: {e}")
            return False
    
    def _normalize_columns(self):
        """Normaliza colunas para formato padrão"""
        # Mapear colunas do dataset completo se necessário
        column_mapping = {
            'repo_full_name': 'repo',
            'author': 'autor', 
            'review_duration_hours': 'tempo_analise_horas',
            'changed_files': 'num_files',
            'description_length': 'descricao_len',
            'comments_count': 'num_comentarios',
            'participants_count': 'num_participants'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in self.df.columns and new_col not in self.df.columns:
                self.df[new_col] = self.df[old_col]
        
        # Criar colunas ausentes com valores padrão
        if 'num_review_comments' not in self.df.columns:
            self.df['num_review_comments'] = 0
        
        if 'state' not in self.df.columns:
            if 'merged' in self.df.columns:
                self.df['state'] = self.df['merged'].apply(lambda x: 'merged' if x else 'closed')
            else:
                self.df['state'] = 'closed'
        
        # Garantir colunas numéricas
        numeric_cols = ['tempo_analise_horas', 'num_files', 'additions', 'deletions', 'num_participants', 'num_comentarios']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Criar tamanho total
        if 'additions' in self.df.columns and 'deletions' in self.df.columns:
            self.df['tamanho_total'] = self.df['additions'].fillna(0) + self.df['deletions'].fillna(0)
        else:
            self.df['tamanho_total'] = 0
        
        # Limpar dados ausentes em colunas essenciais
        essential_cols = ['tempo_analise_horas', 'num_files', 'repo', 'autor']
        existing_cols = [col for col in essential_cols if col in self.df.columns]
        if existing_cols:
            self.df = self.df.dropna(subset=existing_cols)
    
    def _calculate_stats(self):
        """Calcula estatísticas do dataset"""
        self.stats = {
            'total_prs': len(self.df),
            'repositorios': self.df['repo'].nunique() if 'repo' in self.df.columns else 0,
            'autores': self.df['autor'].nunique() if 'autor' in self.df.columns else 0,
            'tempo_medio': self.df['tempo_analise_horas'].mean() if 'tempo_analise_horas' in self.df.columns else 0,
            'mediana_arquivos': self.df['num_files'].median() if 'num_files' in self.df.columns else 0,
            'total_linhas': self.df['tamanho_total'].sum() if 'tamanho_total' in self.df.columns else 0
        }
    
    def print_summary(self):
        """Imprime resumo do dataset"""
        print(f"\n📈 RESUMO - {self.filename}")
        print("=" * 50)
        print(f"• Total de PRs: {self.stats['total_prs']:,}")
        print(f"• Repositórios únicos: {self.stats['repositorios']}")
        print(f"• Autores únicos: {self.stats['autores']:,}")
        print(f"• Tempo médio análise: {self.stats['tempo_medio']:.1f} horas")
        print(f"• Mediana arquivos/PR: {self.stats['mediana_arquivos']:.0f}")
        print(f"• Total linhas modificadas: {self.stats['total_linhas']:,}")
    
    def generate_plots(self, output_dir):
        """Gera todos os gráficos para o dataset"""
        if self.df is None or len(self.df) == 0:
            print(f"⚠️ Sem dados válidos para {self.filename}")
            return False
        
        dataset_name = self.filename.replace('.csv', '')
        
        try:
            # 1. Arquivos vs Tempo
            self._plot_arquivos_vs_tempo(output_dir, dataset_name)
            
            # 2. Tamanho vs Tempo  
            self._plot_tamanho_vs_tempo(output_dir, dataset_name)
            
            # 3. Top Autores
            self._plot_top_autores(output_dir, dataset_name)
            
            # 4. Distribuição Repos
            self._plot_distribuicao_repos(output_dir, dataset_name)
            
            # 5. Estados PRs (se aplicável)
            if 'state' in self.df.columns:
                self._plot_estados_prs(output_dir, dataset_name)
            
            # 6. Participação
            if 'num_participants' in self.df.columns:
                self._plot_participacao(output_dir, dataset_name)
            
            # 7. Comentários vs Tempo
            if 'num_comentarios' in self.df.columns:
                self._plot_comentarios_vs_tempo(output_dir, dataset_name)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao gerar gráficos para {self.filename}: {e}")
            return False
    
    def _plot_arquivos_vs_tempo(self, output_dir, dataset_name):
        """Gráfico: Arquivos vs Tempo"""
        plt.figure(figsize=(12, 8))
        
        # Filtrar outliers
        q99_files = self.df['num_files'].quantile(0.99)
        q99_time = self.df['tempo_analise_horas'].quantile(0.99)
        
        df_filtered = self.df[
            (self.df['num_files'] <= q99_files) & 
            (self.df['tempo_analise_horas'] <= q99_time)
        ]
        
        plt.scatter(df_filtered['num_files'], df_filtered['tempo_analise_horas'],
                   alpha=0.6, s=50, c='#2E86AB', edgecolors='white', linewidth=0.5)
        
        # Tendência
        if len(df_filtered) > 1:
            z = np.polyfit(df_filtered['num_files'], df_filtered['tempo_analise_horas'], 1)
            p = np.poly1d(z)
            plt.plot(df_filtered['num_files'], p(df_filtered['num_files']), 
                     color='#F18F01', linewidth=2, linestyle='--', label='Tendência')
            plt.legend()
        
        plt.xlabel('Número de Arquivos Modificados', fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontweight='bold')
        plt.title(f'Relação: Arquivos × Tempo de Análise\n{dataset_name} ({len(df_filtered):,} PRs)', 
                 fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'arquivos_vs_tempo_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_tamanho_vs_tempo(self, output_dir, dataset_name):
        """Gráfico: Tamanho vs Tempo"""
        plt.figure(figsize=(12, 8))
        
        # Filtrar outliers
        q95_size = self.df['tamanho_total'].quantile(0.95)
        q95_time = self.df['tempo_analise_horas'].quantile(0.95)
        
        df_filtered = self.df[
            (self.df['tamanho_total'] <= q95_size) & 
            (self.df['tempo_analise_horas'] <= q95_time) &
            (self.df['tamanho_total'] > 0)
        ]
        
        if len(df_filtered) > 0:
            color_col = 'num_participants' if 'num_participants' in df_filtered.columns else None
            
            if color_col and df_filtered[color_col].nunique() > 1:
                scatter = plt.scatter(df_filtered['tamanho_total'], 
                                     df_filtered['tempo_analise_horas'],
                                     c=df_filtered[color_col], cmap='viridis',
                                     alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
                plt.colorbar(scatter, label='Participantes')
            else:
                plt.scatter(df_filtered['tamanho_total'], 
                           df_filtered['tempo_analise_horas'],
                           alpha=0.7, s=60, c='#A23B72',
                           edgecolors='white', linewidth=0.5)
        
        plt.xlabel('Total de Linhas Modificadas', fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontweight='bold') 
        plt.title(f'Relação: Tamanho × Tempo de Análise\n{dataset_name} ({len(df_filtered):,} PRs)', 
                 fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'tamanho_vs_tempo_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_top_autores(self, output_dir, dataset_name):
        """Gráfico: Top Autores"""
        plt.figure(figsize=(15, 10))
        
        author_counts = self.df['autor'].value_counts().head(20)
        
        bars = plt.barh(range(len(author_counts)), author_counts.values, 
                       color=sns.color_palette("viridis", len(author_counts)))
        
        plt.yticks(range(len(author_counts)), author_counts.index, fontsize=10)
        plt.xlabel('Número de Pull Requests', fontweight='bold')
        plt.title(f'Top 20 Autores com Mais PRs\n{dataset_name} (Total: {len(self.df):,} PRs)', 
                 fontweight='bold')
        
        # Valores nas barras
        for i, (bar, value) in enumerate(zip(bars, author_counts.values)):
            plt.text(value + max(author_counts.values) * 0.01, i, str(value), 
                    va='center', ha='left', fontweight='bold', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'top_autores_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_distribuicao_repos(self, output_dir, dataset_name):
        """Gráfico: Distribuição Repos"""
        plt.figure(figsize=(15, 8))
        
        repo_counts = self.df['repo'].value_counts().head(15)
        
        bars = plt.bar(range(len(repo_counts)), repo_counts.values,
                      color=sns.color_palette("Set2", len(repo_counts)))
        
        # Nomes dos repos (só a parte final)
        repo_names = [repo.split('/')[-1] if '/' in repo else repo for repo in repo_counts.index]
        plt.xticks(range(len(repo_counts)), repo_names, rotation=45, ha='right')
        
        plt.ylabel('Número de Pull Requests', fontweight='bold')
        plt.title(f'Top 15 Repositórios por PRs\n{dataset_name}', fontweight='bold')
        
        # Valores nas barras
        for bar, value in zip(bars, repo_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(repo_counts.values) * 0.01, 
                    str(value), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'distribuicao_repos_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_estados_prs(self, output_dir, dataset_name):
        """Gráfico: Estados PRs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        state_counts = self.df['state'].value_counts()
        colors = ['#C73E1D', '#7209B7', '#F18F01'][:len(state_counts)]
        
        # Pizza
        ax1.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Distribuição de Estados dos PRs', fontweight='bold')
        
        # Tempo por estado
        avg_time = self.df.groupby('state')['tempo_analise_horas'].mean()
        bars = ax2.bar(avg_time.index, avg_time.values, color=colors[:len(avg_time)])
        
        ax2.set_ylabel('Tempo Médio de Análise (horas)', fontweight='bold')
        ax2.set_title('Tempo Médio por Estado', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Valores
        for bar, value in zip(bars, avg_time.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_time.values) * 0.01, 
                    f'{value:.0f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        filename = output_dir / f'estados_prs_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_participacao(self, output_dir, dataset_name):
        """Gráfico: Participação"""
        plt.figure(figsize=(12, 8))
        
        max_participants = int(self.df['num_participants'].max())
        bins = range(0, min(max_participants + 2, 20))  # Limitar bins para evitar gráficos muito largos
        
        plt.hist(self.df['num_participants'], bins=bins,
                alpha=0.7, color='#2E86AB', edgecolor='white', linewidth=0.5)
        
        plt.xlabel('Número de Participantes por PR', fontweight='bold')
        plt.ylabel('Frequência de PRs', fontweight='bold')
        plt.title(f'Distribuição de Participação em PRs\n{dataset_name}', fontweight='bold')
        
        # Média
        mean_participants = self.df['num_participants'].mean()
        plt.axvline(mean_participants, color='#F18F01', linestyle='--', linewidth=2,
                   label=f'Média: {mean_participants:.1f}')
        
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'participacao_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")
    
    def _plot_comentarios_vs_tempo(self, output_dir, dataset_name):
        """Gráfico: Comentários vs Tempo"""
        plt.figure(figsize=(12, 8))
        
        # Filtrar outliers
        q95_time = self.df['tempo_analise_horas'].quantile(0.95)
        q95_comments = self.df['num_comentarios'].quantile(0.95)
        
        df_filtered = self.df[
            (self.df['tempo_analise_horas'] <= q95_time) &
            (self.df['num_comentarios'] <= q95_comments)
        ]
        
        plt.scatter(df_filtered['num_comentarios'], 
                   df_filtered['tempo_analise_horas'],
                   alpha=0.6, c='#A23B72', s=50,
                   edgecolors='white', linewidth=0.5)
        
        plt.xlabel('Número de Comentários', fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontweight='bold')
        plt.title(f'Relação: Comentários × Tempo\n{dataset_name} ({len(df_filtered):,} PRs)', 
                 fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_dir / f'comentarios_vs_tempo_{dataset_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 {filename.name}")

def create_comparison_report(analyzers, output_dir):
    """Cria relatório comparativo"""
    report_file = output_dir / "relatorio_comparativo.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Relatório Comparativo - Análise de Pull Requests GitHub\n\n")
        f.write("## Datasets Analisados\n\n")
        
        for analyzer in analyzers:
            f.write(f"### {analyzer.filename}\n")
            f.write(f"- **Total de PRs:** {analyzer.stats['total_prs']:,}\n")
            f.write(f"- **Repositórios únicos:** {analyzer.stats['repositorios']}\n") 
            f.write(f"- **Autores únicos:** {analyzer.stats['autores']:,}\n")
            f.write(f"- **Tempo médio análise:** {analyzer.stats['tempo_medio']:.1f} horas\n")
            f.write(f"- **Mediana arquivos/PR:** {analyzer.stats['mediana_arquivos']:.0f}\n")
            f.write(f"- **Total linhas modificadas:** {analyzer.stats['total_linhas']:,}\n\n")
        
        # Comparação se há mais de um dataset
        if len(analyzers) > 1:
            f.write("## Comparação entre Datasets\n\n")
            f.write("| Métrica | Dataset 1 | Dataset 2 | Diferença |\n")
            f.write("|---------|-----------|-----------|----------|\n")
            
            stats1, stats2 = analyzers[0].stats, analyzers[1].stats
            
            metrics = [
                ('PRs Total', 'total_prs'),
                ('Repositórios', 'repositorios'), 
                ('Autores', 'autores'),
                ('Tempo Médio (h)', 'tempo_medio'),
                ('Linhas Totais', 'total_linhas')
            ]
            
            for label, key in metrics:
                val1, val2 = stats1[key], stats2[key]
                if val1 > 0:
                    diff_pct = ((val2/val1 - 1) * 100)
                    diff_str = f"{diff_pct:+.1f}%"
                else:
                    diff_str = "N/A"
                
                f.write(f"| {label} | {val1:,.0f} | {val2:,.0f} | {diff_str} |\n")
        
        f.write(f"\n## Gráficos Gerados\n\n")
        f.write("Os seguintes gráficos foram gerados para cada dataset:\n\n")
        f.write("1. **Arquivos × Tempo de Análise** - Scatter plot mostrando correlação\n")
        f.write("2. **Tamanho × Tempo de Análise** - Relação entre linhas modificadas e tempo\n") 
        f.write("3. **Top 20 Autores** - Ranking dos contribuidores mais ativos\n")
        f.write("4. **Distribuição por Repositório** - PRs por repositório\n")
        f.write("5. **Estados dos PRs** - Distribuição e tempo médio por estado\n")
        f.write("6. **Participação** - Histograma de participantes por PR\n")
        f.write("7. **Comentários × Tempo** - Correlação entre comentários e tempo\n\n")
        
        f.write("---\n")
        f.write(f"*Relatório gerado automaticamente*\n")
    
    print(f"📋 Relatório salvo: {report_file}")

def main():
    """Função principal"""
    print("🚀 ANÁLISE COMPLETA DE PULL REQUESTS DO GITHUB")
    print("=" * 60)
    print("Processando datasets dos top 200 repositórios mais populares\n")
    
    # Datasets disponíveis
    datasets = ["dataset.csv", "dataset_completo.csv"]
    
    # Criar diretório de saída
    output_dir = Path("analise_final")
    output_dir.mkdir(exist_ok=True)
    
    analyzers = []
    
    # Processar cada dataset
    for dataset in datasets:
        if Path(dataset).exists():
            print(f"🔍 Processando: {dataset}")
            analyzer = DatasetAnalyzer(dataset)
            
            if analyzer.load_data():
                analyzer.print_summary()
                
                print(f"🎨 Gerando gráficos para {dataset}...")
                if analyzer.generate_plots(output_dir):
                    analyzers.append(analyzer)
                    print(f"✅ Gráficos concluídos para {dataset}\n")
                else:
                    print(f"❌ Falha ao gerar gráficos para {dataset}\n")
            else:
                print(f"❌ Falha ao carregar {dataset}\n")
        else:
            print(f"⚠️ Arquivo não encontrado: {dataset}\n")
    
    # Criar relatório comparativo
    if analyzers:
        create_comparison_report(analyzers, output_dir)
    
    # Resumo final
    print("🎉 ANÁLISE FINALIZADA!")
    print(f"📁 Resultados salvos em: {output_dir}/")
    print(f"📊 Datasets processados: {len(analyzers)}")
    
    if analyzers:
        total_prs = sum(a.stats['total_prs'] for a in analyzers)
        total_repos = sum(a.stats['repositorios'] for a in analyzers) 
        print(f"📈 Total geral: {total_prs:,} PRs de {total_repos} repositórios")
    
    print("\n💡 Abra os arquivos .png para visualizar os gráficos!")

if __name__ == "__main__":
    main()