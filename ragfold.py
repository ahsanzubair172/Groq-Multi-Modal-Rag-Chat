import os
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

# folder_path = r"D:\Ahsan\Office\Data"
# pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

# # You must provide the llmsherpa_api_url
# llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)

# documents = []
# for pdf_file in pdf_files:
#     documents.extend(pdf_loader.load_data(pdf_file))

# print(pdf_files)
# print(documents)

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_url = "https://arxiv.org/pdf/1910.13461.pdf"
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
documents = pdf_loader.load_data(pdf_url, timeout =None)
print(documents)