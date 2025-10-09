"""
Análise Completa do Dataset Principal - Pull Requests GitHub
===========================================================

Script especializado para análise detalhada do dataset.csv
Foco na qualidade, robustez e visualizações profissionais.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class GitHubPRAnalyzer:
    """Analisador especializado para o dataset principal de PRs"""
    
    def __init__(self, csv_file="dataset.csv"):
        self.csv_file = csv_file
        self.df = None
        self.stats = {}
        self.output_dir = Path("analise_dataset_principal")
        
    def load_and_validate_data(self):
        """Carrega e valida os dados do CSV principal"""
        try:
            print(f"📊 Carregando {self.csv_file}...")
            
            # Carregar CSV
            self.df = pd.read_csv(self.csv_file)
            
            print(f"✅ Dataset carregado: {len(self.df):,} registros")
            print(f"📋 Colunas encontradas: {list(self.df.columns)}")
            
            # Validar estrutura esperada
            expected_columns = [
                'id', 'repo', 'state', 'tempo_analise_horas', 'descricao_len',
                'num_comentarios', 'num_review_comments', 'autor', 'num_files',
                'additions', 'deletions', 'num_participants'
            ]
            
            missing_cols = [col for col in expected_columns if col not in self.df.columns]
            if missing_cols:
                print(f"⚠️ Colunas ausentes: {missing_cols}")
            
            # Limpeza e preparação dos dados
            self._clean_data()
            
            # Calcular estatísticas
            self._calculate_statistics()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar {self.csv_file}: {e}")
            return False
    
    def _clean_data(self):
        """Limpa e prepara os dados"""
        print("🧹 Limpando dados...")
        
        # Remover duplicatas
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id'])
        if len(self.df) < initial_count:
            print(f"   Removidas {initial_count - len(self.df)} duplicatas")
        
        # Garantir tipos corretos
        numeric_columns = [
            'tempo_analise_horas', 'descricao_len', 'num_comentarios',
            'num_review_comments', 'num_files', 'additions', 'deletions', 'num_participants'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remover registros com valores críticos ausentes
        critical_cols = ['repo', 'autor', 'tempo_analise_horas', 'num_files']
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        if len(self.df) < initial_count:
            print(f"   Removidos {initial_count - len(self.df)} registros com dados críticos ausentes")
        
        # Criar colunas derivadas
        self.df['tamanho_total'] = self.df['additions'].fillna(0) + self.df['deletions'].fillna(0)
        self.df['total_comentarios'] = self.df['num_comentarios'].fillna(0) + self.df['num_review_comments'].fillna(0)
        
        # Filtrar outliers extremos (manter 99% dos dados)
        for col in ['tempo_analise_horas', 'num_files', 'tamanho_total']:
            if col in self.df.columns:
                q99 = self.df[col].quantile(0.99)
                outliers_before = len(self.df[self.df[col] > q99])
                self.df = self.df[self.df[col] <= q99]
                if outliers_before > 0:
                    print(f"   Removidos {outliers_before} outliers extremos de {col}")
        
        print(f"✅ Dados limpos: {len(self.df):,} registros finais")
    
    def _calculate_statistics(self):
        """Calcula estatísticas descritivas completas"""
        print("📈 Calculando estatísticas...")
        
        self.stats = {
            # Básicas
            'total_prs': len(self.df),
            'repositorios_unicos': self.df['repo'].nunique(),
            'autores_unicos': self.df['autor'].nunique(),
            
            # Tempo de análise
            'tempo_medio_horas': self.df['tempo_analise_horas'].mean(),
            'tempo_mediano_horas': self.df['tempo_analise_horas'].median(),
            'tempo_std_horas': self.df['tempo_analise_horas'].std(),
            'tempo_min_horas': self.df['tempo_analise_horas'].min(),
            'tempo_max_horas': self.df['tempo_analise_horas'].max(),
            
            # Arquivos
            'arquivos_medio': self.df['num_files'].mean(),
            'arquivos_mediano': self.df['num_files'].median(),
            'arquivos_max': self.df['num_files'].max(),
            
            # Tamanho
            'linhas_total': self.df['tamanho_total'].sum(),
            'linhas_medio': self.df['tamanho_total'].mean(),
            'additions_total': self.df['additions'].sum(),
            'deletions_total': self.df['deletions'].sum(),
            
            # Participação
            'participantes_medio': self.df['num_participants'].mean(),
            'comentarios_medio': self.df['total_comentarios'].mean(),
            
            # Estados
            'estados_distribuicao': self.df['state'].value_counts().to_dict(),
            
            # Top repositórios e autores
            'top_repos': self.df['repo'].value_counts().head(10).to_dict(),
            'top_autores': self.df['autor'].value_counts().head(10).to_dict()
        }
    
    def print_summary(self):
        """Imprime resumo estatístico completo"""
        print("\n" + "="*60)
        print("📊 RESUMO ESTATÍSTICO - DATASET PRINCIPAL")
        print("="*60)
        
        print(f"\n📁 DADOS GERAIS:")
        print(f"   • Total de Pull Requests: {self.stats['total_prs']:,}")
        print(f"   • Repositórios únicos: {self.stats['repositorios_unicos']:,}")
        print(f"   • Autores únicos: {self.stats['autores_unicos']:,}")
        
        print(f"\n⏱️ TEMPO DE ANÁLISE:")
        print(f"   • Média: {self.stats['tempo_medio_horas']:.2f} horas")
        print(f"   • Mediana: {self.stats['tempo_mediano_horas']:.2f} horas")
        print(f"   • Desvio padrão: {self.stats['tempo_std_horas']:.2f} horas")
        print(f"   • Mínimo: {self.stats['tempo_min_horas']:.2f} horas")
        print(f"   • Máximo: {self.stats['tempo_max_horas']:.2f} horas")
        
        print(f"\n📁 ARQUIVOS MODIFICADOS:")
        print(f"   • Média por PR: {self.stats['arquivos_medio']:.1f} arquivos")
        print(f"   • Mediana por PR: {self.stats['arquivos_mediano']:.0f} arquivos")
        print(f"   • Máximo em um PR: {self.stats['arquivos_max']:,} arquivos")
        
        print(f"\n📝 MUDANÇAS DE CÓDIGO:")
        print(f"   • Total de linhas modificadas: {self.stats['linhas_total']:,}")
        print(f"   • Média por PR: {self.stats['linhas_medio']:.0f} linhas")
        print(f"   • Total adições: {self.stats['additions_total']:,}")
        print(f"   • Total deleções: {self.stats['deletions_total']:,}")
        
        print(f"\n👥 COLABORAÇÃO:")
        print(f"   • Participantes médios por PR: {self.stats['participantes_medio']:.1f}")
        print(f"   • Comentários médios por PR: {self.stats['comentarios_medio']:.1f}")
        
        print(f"\n📊 ESTADOS DOS PRS:")
        for estado, count in self.stats['estados_distribuicao'].items():
            porcentagem = (count / self.stats['total_prs']) * 100
            print(f"   • {estado}: {count:,} ({porcentagem:.1f}%)")
        
        print(f"\n🏆 TOP 5 REPOSITÓRIOS:")
        for i, (repo, count) in enumerate(list(self.stats['top_repos'].items())[:5], 1):
            print(f"   {i}. {repo}: {count:,} PRs")
        
        print(f"\n👑 TOP 5 AUTORES:")
        for i, (autor, count) in enumerate(list(self.stats['top_autores'].items())[:5], 1):
            print(f"   {i}. {autor}: {count:,} PRs")
    
    def create_visualizations(self):
        """Cria todas as visualizações"""
        print(f"\n🎨 Criando visualizações...")
        
        # Criar diretório de saída
        self.output_dir.mkdir(exist_ok=True)
        
        visualizations = [
            ("Tempo vs Arquivos", self._plot_tempo_vs_arquivos),
            ("Tempo vs Tamanho", self._plot_tempo_vs_tamanho),
            ("Distribuição Tempo", self._plot_distribuicao_tempo),
            ("Top Repositórios", self._plot_top_repositorios),
            ("Top Autores", self._plot_top_autores),
            ("Estados dos PRs", self._plot_estados_prs),
            ("Matriz Correlação", self._plot_matriz_correlacao),
            ("Participação", self._plot_participacao),
            ("Comentários vs Tempo", self._plot_comentarios_vs_tempo),
            ("Tamanho dos PRs", self._plot_tamanho_prs)
        ]
        
        for nome, func in visualizations:
            try:
                func()
                print(f"   ✅ {nome}")
            except Exception as e:
                print(f"   ❌ {nome}: {e}")
        
        print(f"\n📁 Gráficos salvos em: {self.output_dir}")
    
    def _plot_tempo_vs_arquivos(self):
        """Gráfico: Tempo vs Número de Arquivos"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot com densidade de cor
        plt.scatter(self.df['num_files'], self.df['tempo_analise_horas'],
                   alpha=0.6, s=50, c='#2E86AB', edgecolors='white', linewidth=0.5)
        
        # Linha de tendência
        z = np.polyfit(self.df['num_files'], self.df['tempo_analise_horas'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['num_files'], p(self.df['num_files']),
                 color='#F18F01', linewidth=2, linestyle='--', 
                 label=f'Tendência (R² = {np.corrcoef(self.df["num_files"], self.df["tempo_analise_horas"])[0,1]**2:.3f})')
        
        # Configurações
        plt.xlabel('Número de Arquivos Modificados', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
        plt.title('Relação entre Arquivos Modificados e Tempo de Análise\n' +
                 f'{len(self.df):,} Pull Requests', fontsize=14, fontweight='bold')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'tempo_vs_arquivos.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tempo_vs_tamanho(self):
        """Gráfico: Tempo vs Tamanho Total"""
        plt.figure(figsize=(12, 8))
        
        # Filtrar PRs com mudanças
        df_com_mudancas = self.df[self.df['tamanho_total'] > 0]
        
        # Scatter plot colorido por participantes
        scatter = plt.scatter(df_com_mudancas['tamanho_total'], 
                             df_com_mudancas['tempo_analise_horas'],
                             c=df_com_mudancas['num_participants'], 
                             cmap='viridis', alpha=0.7, s=60,
                             edgecolors='white', linewidth=0.5)
        
        plt.colorbar(scatter, label='Número de Participantes')
        
        plt.xlabel('Total de Linhas Modificadas', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
        plt.title('Relação entre Tamanho do PR e Tempo de Análise\n' +
                 f'{len(df_com_mudancas):,} PRs com mudanças', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'tempo_vs_tamanho.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribuicao_tempo(self):
        """Gráfico: Distribuição do Tempo de Análise"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(self.df['tempo_analise_horas'], bins=50, alpha=0.7, 
                color='#2E86AB', edgecolor='white')
        
        ax1.axvline(self.stats['tempo_medio_horas'], color='#F18F01', 
                   linestyle='--', linewidth=2, label=f'Média: {self.stats["tempo_medio_horas"]:.1f}h')
        ax1.axvline(self.stats['tempo_mediano_horas'], color='#C73E1D', 
                   linestyle='--', linewidth=2, label=f'Mediana: {self.stats["tempo_mediano_horas"]:.1f}h')
        
        ax1.set_xlabel('Tempo de Análise (horas)', fontweight='bold')
        ax1.set_ylabel('Frequência', fontweight='bold')
        ax1.set_title('Distribuição do Tempo de Análise', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot por estado
        estados_df = [self.df[self.df['state'] == estado]['tempo_analise_horas'] 
                     for estado in self.df['state'].unique()]
        
        bp = ax2.boxplot(estados_df, labels=self.df['state'].unique(), patch_artist=True)
        
        colors = ['#2E86AB', '#F18F01', '#C73E1D'][:len(bp['boxes'])]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Tempo de Análise (horas)', fontweight='bold')
        ax2.set_title('Tempo por Estado do PR', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribuicao_tempo.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_repositorios(self):
        """Gráfico: Top 15 Repositórios"""
        plt.figure(figsize=(15, 10))
        
        top_repos = self.df['repo'].value_counts().head(15)
        
        # Nomes simplificados
        repo_names = [repo.split('/')[-1] if '/' in repo else repo for repo in top_repos.index]
        
        bars = plt.barh(range(len(top_repos)), top_repos.values,
                       color=sns.color_palette("viridis", len(top_repos)))
        
        plt.yticks(range(len(top_repos)), repo_names)
        plt.xlabel('Número de Pull Requests', fontsize=12, fontweight='bold')
        plt.title('Top 15 Repositórios com Mais Pull Requests\n' +
                 f'Total: {len(self.df):,} PRs', fontsize=14, fontweight='bold')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, top_repos.values)):
            plt.text(value + max(top_repos.values) * 0.01, i, str(value),
                    va='center', ha='left', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'top_repositorios.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_autores(self):
        """Gráfico: Top 20 Autores"""
        plt.figure(figsize=(15, 10))
        
        top_autores = self.df['autor'].value_counts().head(20)
        
        bars = plt.barh(range(len(top_autores)), top_autores.values,
                       color=sns.color_palette("plasma", len(top_autores)))
        
        plt.yticks(range(len(top_autores)), top_autores.index)
        plt.xlabel('Número de Pull Requests', fontsize=12, fontweight='bold')
        plt.title('Top 20 Autores com Mais Pull Requests\n' +
                 f'Total: {self.stats["autores_unicos"]:,} autores únicos', 
                 fontsize=14, fontweight='bold')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, top_autores.values)):
            plt.text(value + max(top_autores.values) * 0.01, i, str(value),
                    va='center', ha='left', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'top_autores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_estados_prs(self):
        """Gráfico: Estados dos PRs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pizza - Distribuição
        state_counts = self.df['state'].value_counts()
        colors = ['#2E86AB', '#F18F01', '#C73E1D'][:len(state_counts)]
        
        wedges, texts, autotexts = ax1.pie(state_counts.values, labels=state_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax1.set_title('Distribuição dos Estados dos PRs', fontweight='bold')
        
        # Barra - Tempo médio por estado
        avg_time_by_state = self.df.groupby('state')['tempo_analise_horas'].mean()
        
        bars = ax2.bar(avg_time_by_state.index, avg_time_by_state.values,
                      color=colors[:len(avg_time_by_state)])
        
        ax2.set_ylabel('Tempo Médio de Análise (horas)', fontweight='bold')
        ax2.set_title('Tempo Médio por Estado', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Valores nas barras
        for bar, value in zip(bars, avg_time_by_state.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_time_by_state.values) * 0.01,
                    f'{value:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'estados_prs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_matriz_correlacao(self):
        """Gráfico: Matriz de Correlação"""
        plt.figure(figsize=(12, 10))
        
        # Selecionar variáveis numéricas
        numeric_cols = [
            'tempo_analise_horas', 'descricao_len', 'num_comentarios',
            'num_review_comments', 'num_files', 'additions', 'deletions',
            'num_participants', 'tamanho_total', 'total_comentarios'
        ]
        
        # Filtrar colunas existentes
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        
        # Criar heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Correlação - Variáveis dos PRs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_participacao(self):
        """Gráfico: Distribuição de Participantes"""
        plt.figure(figsize=(12, 8))
        
        # Limitar a 15 participantes para visualização
        max_participants = min(self.df['num_participants'].max(), 15)
        bins = range(0, max_participants + 2)
        
        plt.hist(self.df['num_participants'], bins=bins, alpha=0.7,
                color='#2E86AB', edgecolor='white', linewidth=1)
        
        # Linha da média
        mean_participants = self.df['num_participants'].mean()
        plt.axvline(mean_participants, color='#F18F01', linestyle='--', linewidth=2,
                   label=f'Média: {mean_participants:.1f}')
        
        plt.xlabel('Número de Participantes por PR', fontsize=12, fontweight='bold')
        plt.ylabel('Frequência de PRs', fontsize=12, fontweight='bold')
        plt.title('Distribuição de Participação em Pull Requests\n' +
                 f'Média: {mean_participants:.1f} participantes por PR', 
                 fontsize=14, fontweight='bold')
        
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'participacao.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comentarios_vs_tempo(self):
        """Gráfico: Comentários vs Tempo"""
        plt.figure(figsize=(12, 8))
        
        plt.scatter(self.df['total_comentarios'], self.df['tempo_analise_horas'],
                   alpha=0.6, c='#A23B72', s=50, edgecolors='white', linewidth=0.5)
        
        # Linha de tendência se houver correlação
        if self.df['total_comentarios'].nunique() > 1:
            z = np.polyfit(self.df['total_comentarios'], self.df['tempo_analise_horas'], 1)
            p = np.poly1d(z)
            plt.plot(self.df['total_comentarios'], p(self.df['total_comentarios']),
                     color='#F18F01', linewidth=2, linestyle='--', label='Tendência')
            plt.legend()
        
        plt.xlabel('Total de Comentários (Review + Discussão)', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=12, fontweight='bold')
        plt.title('Relação entre Comentários e Tempo de Análise\n' +
                 f'{len(self.df):,} Pull Requests', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'comentarios_vs_tempo.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tamanho_prs(self):
        """Gráfico: Distribuição do Tamanho dos PRs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma de arquivos
        ax1.hist(self.df['num_files'], bins=30, alpha=0.7,
                color='#2E86AB', edgecolor='white')
        
        ax1.axvline(self.stats['arquivos_medio'], color='#F18F01', 
                   linestyle='--', linewidth=2, label=f'Média: {self.stats["arquivos_medio"]:.1f}')
        ax1.axvline(self.stats['arquivos_mediano'], color='#C73E1D', 
                   linestyle='--', linewidth=2, label=f'Mediana: {self.stats["arquivos_mediano"]:.0f}')
        
        ax1.set_xlabel('Número de Arquivos Modificados', fontweight='bold')
        ax1.set_ylabel('Frequência', fontweight='bold')
        ax1.set_title('Distribuição: Arquivos por PR', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma de linhas (log scale para melhor visualização)
        df_com_mudancas = self.df[self.df['tamanho_total'] > 0]
        
        ax2.hist(df_com_mudancas['tamanho_total'], bins=50, alpha=0.7,
                color='#A23B72', edgecolor='white')
        
        ax2.set_xlabel('Total de Linhas Modificadas', fontweight='bold')
        ax2.set_ylabel('Frequência', fontweight='bold')
        ax2.set_title('Distribuição: Linhas por PR', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tamanho_prs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Gera relatório em markdown"""
        report_file = self.output_dir / "relatorio_dataset_principal.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Análise - Dataset Principal de Pull Requests\n\n")
            f.write(f"**Data da Análise:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"**Arquivo Analisado:** {self.csv_file}\n\n")
            
            f.write("## Resumo Executivo\n\n")
            f.write(f"Este relatório apresenta a análise completa de **{self.stats['total_prs']:,} Pull Requests** ")
            f.write(f"de **{self.stats['repositorios_unicos']}** repositórios únicos no GitHub, ")
            f.write(f"criados por **{self.stats['autores_unicos']:,}** desenvolvedores diferentes.\n\n")
            
            f.write("## Estatísticas Principais\n\n")
            f.write("### Tempo de Análise\n")
            f.write(f"- **Tempo médio:** {self.stats['tempo_medio_horas']:.2f} horas\n")
            f.write(f"- **Tempo mediano:** {self.stats['tempo_mediano_horas']:.2f} horas\n")
            f.write(f"- **Desvio padrão:** {self.stats['tempo_std_horas']:.2f} horas\n")
            f.write(f"- **Intervalo:** {self.stats['tempo_min_horas']:.2f} - {self.stats['tempo_max_horas']:.2f} horas\n\n")
            
            f.write("### Modificações de Código\n")
            f.write(f"- **Arquivos médios por PR:** {self.stats['arquivos_medio']:.1f}\n")
            f.write(f"- **Linhas totais modificadas:** {self.stats['linhas_total']:,}\n")
            f.write(f"- **Linhas médias por PR:** {self.stats['linhas_medio']:.0f}\n")
            f.write(f"- **Total de adições:** {self.stats['additions_total']:,}\n")
            f.write(f"- **Total de deleções:** {self.stats['deletions_total']:,}\n\n")
            
            f.write("### Colaboração\n")
            f.write(f"- **Participantes médios:** {self.stats['participantes_medio']:.1f} por PR\n")
            f.write(f"- **Comentários médios:** {self.stats['comentarios_medio']:.1f} por PR\n\n")
            
            f.write("### Estados dos Pull Requests\n")
            for estado, count in self.stats['estados_distribuicao'].items():
                porcentagem = (count / self.stats['total_prs']) * 100
                f.write(f"- **{estado}:** {count:,} ({porcentagem:.1f}%)\n")
            f.write("\n")
            
            f.write("## Top Repositórios\n\n")
            for i, (repo, count) in enumerate(list(self.stats['top_repos'].items())[:10], 1):
                f.write(f"{i}. **{repo}:** {count:,} PRs\n")
            f.write("\n")
            
            f.write("## Top Autores\n\n")
            for i, (autor, count) in enumerate(list(self.stats['top_autores'].items())[:10], 1):
                f.write(f"{i}. **{autor}:** {count:,} PRs\n")
            f.write("\n")
            
            f.write("## Gráficos Gerados\n\n")
            graficos = [
                "tempo_vs_arquivos.png - Relação entre número de arquivos e tempo",
                "tempo_vs_tamanho.png - Relação entre tamanho do PR e tempo",
                "distribuicao_tempo.png - Distribuição e box plots do tempo",
                "top_repositorios.png - Ranking dos repositórios",
                "top_autores.png - Ranking dos autores",
                "estados_prs.png - Distribuição e tempo por estado",
                "matriz_correlacao.png - Correlações entre variáveis",
                "participacao.png - Distribuição de participantes",
                "comentarios_vs_tempo.png - Relação comentários e tempo",
                "tamanho_prs.png - Distribuição de arquivos e linhas"
            ]
            
            for grafico in graficos:
                f.write(f"- {grafico}\n")
            
            f.write(f"\n---\n*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}*")
        
        print(f"📋 Relatório salvo: {report_file}")

def main():
    """Função principal"""
    print("🚀 ANÁLISE COMPLETA DO DATASET PRINCIPAL")
    print("=" * 50)
    print("GitHub Pull Requests - Top 200 Repositórios\n")
    
    # Inicializar analisador
    analyzer = GitHubPRAnalyzer("dataset.csv")
    
    # Carregar e validar dados
    if not analyzer.load_and_validate_data():
        print("❌ Falha ao carregar dados. Encerrando.")
        return
    
    # Mostrar resumo estatístico
    analyzer.print_summary()
    
    # Criar visualizações
    analyzer.create_visualizations()
    
    # Gerar relatório
    analyzer.generate_report()
    
    # Resumo final
    print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"📁 Resultados salvos em: {analyzer.output_dir}/")
    print(f"📊 {analyzer.stats['total_prs']:,} PRs analisados")
    print(f"📈 10 gráficos profissionais gerados")
    print(f"📋 Relatório completo em markdown")
    print(f"\n💡 Abra os arquivos .png para visualizar os insights!")

if __name__ == "__main__":
    main()