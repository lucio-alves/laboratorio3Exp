"""
Análise Dataset Completo - 3 Gráficos Específicos
=================================================

Script para gerar 3 gráficos específicos do dataset_completo.csv:
1. Relação: Número de Arquivos × Tempo de Análise (scatter plot)
2. Relação: Tamanho Total (Linhas) × Tempo de Análise (scatter plot)
3. Top 20 Autores com Mais PRs no Dataset (bar chart horizontal)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DatasetCompletoAnalyzer:
    """Analisador para gerar 3 gráficos específicos do dataset completo"""
    
    def __init__(self, csv_file="dataset_completo.csv"):
        self.csv_file = csv_file
        self.df = None
        self.output_dir = Path("graficos_dataset_completo")
        
    def load_data(self):
        """Carrega e prepara os dados"""
        try:
            print(f"📊 Carregando {self.csv_file}...")
            
            # Carregar CSV
            self.df = pd.read_csv(self.csv_file)
            
            print(f"✅ Dataset carregado: {len(self.df):,} registros")
            print(f"📋 Colunas: {list(self.df.columns)}")
            
            # Preparar dados
            self._prepare_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar {self.csv_file}: {e}")
            return False
    
    def _prepare_data(self):
        """Prepara e limpa os dados"""
        print("🧹 Preparando dados...")
        
        # Mapear colunas para nomes padronizados
        column_mapping = {
            'repo_full_name': 'repo',
            'review_duration_hours': 'tempo_analise_horas',
            'changed_files': 'num_files',
            'participants_count': 'num_participants',
            'comments_count': 'num_comentarios'
        }
        
        # Renomear colunas
        for old_col, new_col in column_mapping.items():
            if old_col in self.df.columns:
                self.df[new_col] = self.df[old_col]
        
        # Garantir tipos numéricos
        numeric_cols = ['tempo_analise_horas', 'num_files', 'additions', 'deletions']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Criar coluna de tamanho total
        self.df['tamanho_total'] = self.df['additions'].fillna(0) + self.df['deletions'].fillna(0)
        
        # Remover registros com dados críticos ausentes
        critical_cols = ['author', 'tempo_analise_horas', 'num_files']
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        
        if len(self.df) < initial_count:
            print(f"   Removidos {initial_count - len(self.df)} registros com dados ausentes")
        
        # Filtrar outliers extremos (manter 99% dos dados para melhor visualização)
        for col in ['tempo_analise_horas', 'num_files', 'tamanho_total']:
            if col in self.df.columns and len(self.df) > 0:
                q99 = self.df[col].quantile(0.99)
                outliers_before = len(self.df[self.df[col] > q99])
                self.df = self.df[self.df[col] <= q99]
                if outliers_before > 0:
                    print(f"   Filtrados {outliers_before} outliers extremos de {col}")
        
        print(f"✅ Dados preparados: {len(self.df):,} registros finais")
        
        # Estatísticas básicas
        if len(self.df) > 0:
            print(f"📊 Repositórios únicos: {self.df['repo'].nunique() if 'repo' in self.df.columns else 'N/A'}")
            print(f"👥 Autores únicos: {self.df['author'].nunique()}")
            print(f"⏱️ Tempo médio análise: {self.df['tempo_analise_horas'].mean():.1f} horas")
    
    def create_graphs(self):
        """Cria os 3 gráficos específicos"""
        if self.df is None or len(self.df) == 0:
            print("❌ Sem dados para gerar gráficos")
            return False
        
        # Criar diretório de saída
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎨 Gerando 3 gráficos específicos...")
        
        try:
            # 1. Arquivos vs Tempo
            self._plot_arquivos_vs_tempo()
            print("   ✅ Gráfico 1: Arquivos × Tempo")
            
            # 2. Tamanho vs Tempo
            self._plot_tamanho_vs_tempo()
            print("   ✅ Gráfico 2: Tamanho × Tempo")
            
            # 3. Top Autores
            self._plot_top_autores()
            print("   ✅ Gráfico 3: Top 20 Autores")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao gerar gráficos: {e}")
            return False
    
    def _plot_arquivos_vs_tempo(self):
        """Gráfico 1: Relação Número de Arquivos × Tempo de Análise"""
        plt.figure(figsize=(14, 10))
        
        # Scatter plot com transparência
        scatter = plt.scatter(
            self.df['num_files'], 
            self.df['tempo_analise_horas'],
            alpha=0.6, 
            s=60, 
            c='#2E86AB', 
            edgecolors='white', 
            linewidth=0.5
        )
        
        # Linha de tendência
        if len(self.df) > 1:
            try:
                z = np.polyfit(self.df['num_files'], self.df['tempo_analise_horas'], 1)
                p = np.poly1d(z)
                plt.plot(self.df['num_files'], p(self.df['num_files']), 
                         color='#F18F01', linewidth=3, linestyle='--', 
                         label='Linha de Tendência')
                
                # Coeficiente de correlação
                corr = np.corrcoef(self.df['num_files'], self.df['tempo_analise_horas'])[0,1]
                plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', 
                        transform=plt.gca().transAxes, 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.legend(fontsize=12)
                
            except Exception as e:
                print(f"   ⚠️ Aviso: Não foi possível calcular tendência: {e}")
        
        # Configurações do gráfico
        plt.xlabel('Número de Arquivos Modificados', fontsize=14, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=14, fontweight='bold')
        plt.title('Relação: Número de Arquivos × Tempo de Análise\n' +
                 f'Dataset Completo ({len(self.df):,} Pull Requests)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Grid e formatação
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=11)
        
        # Anotação com estatísticas
        stats_text = f'Total PRs: {len(self.df):,}\n'
        stats_text += f'Arquivos médios: {self.df["num_files"].mean():.1f}\n'
        stats_text += f'Tempo médio: {self.df["tempo_analise_horas"].mean():.1f}h'
        
        plt.text(0.95, 0.05, stats_text, 
                transform=plt.gca().transAxes, 
                fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'arquivos_vs_tempo_completo.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_tamanho_vs_tempo(self):
        """Gráfico 2: Relação Tamanho Total × Tempo de Análise"""
        plt.figure(figsize=(14, 10))
        
        # Filtrar PRs com mudanças
        df_com_mudancas = self.df[self.df['tamanho_total'] > 0].copy()
        
        if len(df_com_mudancas) == 0:
            print("   ⚠️ Sem dados de mudanças para o gráfico 2")
            return
        
        # Scatter plot colorido por número de arquivos (se disponível)
        if 'num_files' in df_com_mudancas.columns and df_com_mudancas['num_files'].nunique() > 1:
            scatter = plt.scatter(
                df_com_mudancas['tamanho_total'], 
                df_com_mudancas['tempo_analise_horas'],
                c=df_com_mudancas['num_files'], 
                cmap='viridis', 
                alpha=0.7, 
                s=60,
                edgecolors='white', 
                linewidth=0.5
            )
            plt.colorbar(scatter, label='Número de Arquivos')
        else:
            plt.scatter(
                df_com_mudancas['tamanho_total'], 
                df_com_mudancas['tempo_analise_horas'],
                alpha=0.7, 
                s=60, 
                c='#A23B72',
                edgecolors='white', 
                linewidth=0.5
            )
        
        # Linha de tendência
        if len(df_com_mudancas) > 1:
            try:
                z = np.polyfit(df_com_mudancas['tamanho_total'], df_com_mudancas['tempo_analise_horas'], 1)
                p = np.poly1d(z)
                plt.plot(df_com_mudancas['tamanho_total'], p(df_com_mudancas['tamanho_total']),
                         color='#F18F01', linewidth=3, linestyle='--',
                         label='Linha de Tendência')
                
                # Coeficiente de correlação
                corr = np.corrcoef(df_com_mudancas['tamanho_total'], df_com_mudancas['tempo_analise_horas'])[0,1]
                plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', 
                        transform=plt.gca().transAxes, 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.legend(fontsize=12)
                
            except Exception as e:
                print(f"   ⚠️ Aviso: Não foi possível calcular tendência: {e}")
        
        # Configurações do gráfico
        plt.xlabel('Total de Linhas Modificadas (Adições + Deleções)', fontsize=14, fontweight='bold')
        plt.ylabel('Tempo de Análise (horas)', fontsize=14, fontweight='bold')
        plt.title('Relação: Tamanho Total (Linhas) × Tempo de Análise\n' +
                 f'Dataset Completo ({len(df_com_mudancas):,} PRs com mudanças)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Grid e formatação
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=11)
        
        # Anotação com estatísticas
        stats_text = f'PRs com mudanças: {len(df_com_mudancas):,}\n'
        stats_text += f'Linhas médias: {df_com_mudancas["tamanho_total"].mean():.0f}\n'
        stats_text += f'Tempo médio: {df_com_mudancas["tempo_analise_horas"].mean():.1f}h'
        
        plt.text(0.95, 0.05, stats_text, 
                transform=plt.gca().transAxes, 
                fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tamanho_vs_tempo_completo.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_top_autores(self):
        """Gráfico 3: Top 20 Autores com Mais PRs"""
        plt.figure(figsize=(16, 12))
        
        # Calcular top 20 autores
        top_autores = self.df['author'].value_counts().head(20)
        
        if len(top_autores) == 0:
            print("   ⚠️ Sem dados de autores para o gráfico 3")
            return
        
        # Criar paleta de cores
        colors = sns.color_palette("viridis", len(top_autores))
        
        # Gráfico de barras horizontal
        bars = plt.barh(range(len(top_autores)), top_autores.values, color=colors)
        
        # Configurar eixos
        plt.yticks(range(len(top_autores)), top_autores.index, fontsize=11)
        plt.xlabel('Número de Pull Requests', fontsize=14, fontweight='bold')
        plt.title('Top 20 Autores com Mais PRs no Dataset Completo\n' +
                 f'Total: {len(self.df):,} PRs de {self.df["author"].nunique():,} autores únicos', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, top_autores.values)):
            plt.text(value + max(top_autores.values) * 0.01, i, str(value),
                    va='center', ha='left', fontweight='bold', fontsize=10)
        
        # Destacar top 3
        for i in range(min(3, len(bars))):
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(2)
        
        # Grid e formatação
        plt.grid(axis='x', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=11)
        
        # Inverter ordem do eixo Y (maior no topo)
        plt.gca().invert_yaxis()
        
        # Anotação com estatísticas
        total_prs_top20 = top_autores.sum()
        percentage_top20 = (total_prs_top20 / len(self.df)) * 100
        
        stats_text = f'Top 20 autores:\n'
        stats_text += f'{total_prs_top20:,} PRs ({percentage_top20:.1f}%)\n'
        stats_text += f'Média: {top_autores.mean():.1f} PRs/autor'
        
        plt.text(0.95, 0.95, stats_text, 
                transform=plt.gca().transAxes, 
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_autores_completo.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def print_summary(self):
        """Imprime resumo dos dados"""
        if self.df is None or len(self.df) == 0:
            return
        
        print(f"\n📊 RESUMO DO DATASET COMPLETO")
        print("=" * 40)
        print(f"📁 Total de PRs analisados: {len(self.df):,}")
        print(f"🏢 Repositórios únicos: {self.df['repo'].nunique() if 'repo' in self.df.columns else 'N/A'}")
        print(f"👥 Autores únicos: {self.df['author'].nunique()}")
        print(f"⏱️ Tempo médio de análise: {self.df['tempo_analise_horas'].mean():.1f} horas")
        print(f"📝 Arquivos médios por PR: {self.df['num_files'].mean():.1f}")
        print(f"📊 Linhas médias por PR: {self.df['tamanho_total'].mean():.0f}")
        
        # Top 3 autores
        top3 = self.df['author'].value_counts().head(3)
        print(f"\n🏆 TOP 3 AUTORES:")
        for i, (autor, count) in enumerate(top3.items(), 1):
            print(f"   {i}. {autor}: {count} PRs")

def main():
    """Função principal"""
    print("🚀 ANÁLISE DATASET COMPLETO - 3 GRÁFICOS ESPECÍFICOS")
    print("=" * 60)
    print("Gerando visualizações focadas dos Pull Requests\n")
    
    # Inicializar analisador
    analyzer = DatasetCompletoAnalyzer("dataset_completo.csv")
    
    # Carregar dados
    if not analyzer.load_data():
        print("❌ Falha ao carregar dados. Encerrando.")
        return
    
    # Mostrar resumo
    analyzer.print_summary()
    
    # Criar gráficos
    if analyzer.create_graphs():
        print(f"\n🎉 ANÁLISE CONCLUÍDA!")
        print(f"📁 Gráficos salvos em: {analyzer.output_dir}/")
        print(f"📊 Dataset: {len(analyzer.df):,} PRs processados")
        print(f"\n🎨 GRÁFICOS GERADOS:")
        print(f"   1. arquivos_vs_tempo_completo.png")
        print(f"   2. tamanho_vs_tempo_completo.png") 
        print(f"   3. top_autores_completo.png")
        print(f"\n💡 Abra os arquivos .png para visualizar!")
    else:
        print("❌ Falha ao gerar gráficos.")

if __name__ == "__main__":
    main()