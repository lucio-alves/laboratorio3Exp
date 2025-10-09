"""
Conversor de arquivos XLSX para CSV otimizado para arquivos grandes
"""

import pandas as pd
import os
import sys
from pathlib import Path
import argparse
import time

def convert_xlsx_to_csv(input_file, output_file=None, sheet_name=None, chunk_size=10000):
    """
    Converte arquivo XLSX para CSV com suporte a arquivos grandes
    
    Args:
        input_file (str): Caminho do arquivo XLSX de entrada
        output_file (str): Caminho do arquivo CSV de saída (opcional)
        sheet_name (str): Nome da planilha específica (opcional)
        chunk_size (int): Tamanho do chunk para processamento em lotes
    """
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo {input_file} não encontrado!")
        return False
    
    # Gerar nome do arquivo de saída se não especificado
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.csv')
    
    print(f"Iniciando conversão de {input_file} para {output_file}")
    
    try:
        # Verificar informações do arquivo XLSX
        excel_file = pd.ExcelFile(input_file)
        print(f"Planilhas disponíveis: {excel_file.sheet_names}")
        
        # Determinar qual planilha usar
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]
            print(f"Usando a primeira planilha: {sheet_name}")
        elif sheet_name not in excel_file.sheet_names:
            print(f"Erro: Planilha '{sheet_name}' não encontrada!")
            return False
        
        # Obter informações sobre o tamanho dos dados
        print(f"Processando planilha: {sheet_name}")
        
        start_time = time.time()
        
        # Método 1: Para arquivos menores (carregamento direto)
        try:
            print("Tentando carregamento direto...")
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            
            # Salvar como CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            rows, cols = df.shape
            print(f"Conversão concluída com sucesso!")
            print(f"Dados processados: {rows:,} linhas x {cols} colunas")
            
        except MemoryError:
            print("Arquivo muito grande para carregamento direto. Usando processamento em chunks...")
            
            # Método 2: Processamento em chunks para arquivos grandes
            first_chunk = True
            total_rows = 0
            
            # Usar chunksize com pd.read_excel (limitado)
            for chunk_num, chunk in enumerate(pd.read_excel(input_file, 
                                                           sheet_name=sheet_name, 
                                                           chunksize=chunk_size), 1):
                
                # Primeira iteração: criar arquivo e cabeçalho
                if first_chunk:
                    chunk.to_csv(output_file, index=False, encoding='utf-8')
                    first_chunk = False
                    cols = chunk.shape[1]
                else:
                    # Anexar sem cabeçalho
                    chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
                
                total_rows += len(chunk)
                print(f"Processado chunk {chunk_num}: {len(chunk):,} linhas")
            
            print(f"Conversão concluída com processamento em chunks!")
            print(f"Dados processados: {total_rows:,} linhas x {cols} colunas")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Tempo de processamento: {processing_time:.2f} segundos")
        
        # Verificar tamanho dos arquivos
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"Arquivo original: {input_size:.2f} MB")
        print(f"Arquivo CSV: {output_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Erro durante a conversão: {str(e)}")
        return False

def list_xlsx_files(directory="."):
    """Lista todos os arquivos XLSX no diretório especificado"""
    xlsx_files = []
    for file in Path(directory).glob("*.xlsx"):
        xlsx_files.append(str(file))
    return xlsx_files

def main():
    parser = argparse.ArgumentParser(description='Conversor de XLSX para CSV')
    parser.add_argument('input_file', nargs='?', help='Arquivo XLSX de entrada')
    parser.add_argument('-o', '--output', help='Arquivo CSV de saída')
    parser.add_argument('-s', '--sheet', help='Nome da planilha específica')
    parser.add_argument('-c', '--chunk-size', type=int, default=10000, 
                       help='Tamanho do chunk para arquivos grandes (padrão: 10000)')
    parser.add_argument('-l', '--list', action='store_true', 
                       help='Listar arquivos XLSX disponíveis')
    
    args = parser.parse_args()
    
    # Listar arquivos XLSX disponíveis
    if args.list:
        xlsx_files = list_xlsx_files()
        if xlsx_files:
            print("Arquivos XLSX encontrados:")
            for i, file in enumerate(xlsx_files, 1):
                print(f"{i}. {file}")
        else:
            print("Nenhum arquivo XLSX encontrado no diretório atual.")
        return
    
    # Se não especificou arquivo, mostrar arquivos disponíveis
    if not args.input_file:
        xlsx_files = list_xlsx_files()
        if xlsx_files:
            print("Arquivos XLSX disponíveis:")
            for i, file in enumerate(xlsx_files, 1):
                print(f"{i}. {file}")
            
            try:
                choice = input("\nEscolha um arquivo (número) ou digite o caminho: ")
                if choice.isdigit() and 1 <= int(choice) <= len(xlsx_files):
                    args.input_file = xlsx_files[int(choice) - 1]
                else:
                    args.input_file = choice
            except KeyboardInterrupt:
                print("\nOperação cancelada.")
                return
        else:
            print("Nenhum arquivo XLSX encontrado. Use: python converter_xlsx_to_csv.py <arquivo.xlsx>")
            return
    
    # Realizar a conversão
    success = convert_xlsx_to_csv(
        input_file=args.input_file,
        output_file=args.output,
        sheet_name=args.sheet,
        chunk_size=args.chunk_size
    )
    
    if success:
        print("Conversão realizada com sucesso!")
    else:
        print("Falha na conversão.")
        sys.exit(1)

if __name__ == "__main__":
    main()