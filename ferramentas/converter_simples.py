"""
Script simples para converter XLSX para CSV
"""

import pandas as pd
import os

def converter_xlsx_para_csv(arquivo_xlsx, arquivo_csv=None):
    """
    Converte um arquivo XLSX para CSV
    
    Args:
        arquivo_xlsx (str): Caminho do arquivo XLSX
        arquivo_csv (str): Caminho do arquivo CSV de saída (opcional)
    """
    
    # Verificar se arquivo existe
    if not os.path.exists(arquivo_xlsx):
        print(f"Erro: Arquivo {arquivo_xlsx} não encontrado!")
        return
    
    # Gerar nome do CSV se não especificado
    if arquivo_csv is None:
        arquivo_csv = arquivo_xlsx.replace('.xlsx', '.csv')
    
    try:
        print(f"Carregando arquivo: {arquivo_xlsx}")
        
        # Carregar o arquivo Excel
        df = pd.read_excel(arquivo_xlsx)
        
        print(f"Dados carregados: {len(df):,} linhas x {len(df.columns)} colunas")
        
        # Salvar como CSV
        df.to_csv(arquivo_csv, index=False, encoding='utf-8')
        
        print(f"Arquivo salvo como: {arquivo_csv}")
        print("Conversão concluída com sucesso!")
        
        # Mostrar informações dos arquivos
        tamanho_xlsx = os.path.getsize(arquivo_xlsx) / (1024 * 1024)  # MB
        tamanho_csv = os.path.getsize(arquivo_csv) / (1024 * 1024)   # MB
        
        print(f"\nInformações:")
        print(f"Arquivo XLSX: {tamanho_xlsx:.2f} MB")
        print(f"Arquivo CSV: {tamanho_csv:.2f} MB")
        
    except Exception as e:
        print(f"Erro durante a conversão: {e}")

# Exemplo de uso
if __name__ == "__main__":
    # Converter o arquivo dataset_completo.xlsx que está no seu workspace
    converter_xlsx_para_csv("dataset_completo.xlsx")
    
    # Ou converter o top200_repositorios.xlsx
    # converter_xlsx_para_csv("top200_repositorios.xlsx")
    
    # Você também pode especificar um nome personalizado para o CSV:
    # converter_xlsx_para_csv("dataset_completo.xlsx", "meu_dataset.csv")