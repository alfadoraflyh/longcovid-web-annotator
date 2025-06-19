# longcovid-web-annotator
## Deskripsi
Aplikasi **LongCOVID Web Annotator** ini menyediakan antarmuka berbasis web untuk menganotasi gejala Long COVID pada teks input. Prosesnya dua tahap: klasifikasi teks menggunakan model BERT untuk mendeteksi gejala, lalu anotasi gejala dengan RAG berbasis Knowledge Graph.

## Fitur
- **Antarmuka Web** menggunakan Flask   
- **Klasifikasi Teks** dengan model BERT (fine-tuned) via Transformers  
- **Retrieval-Augmented Generation (RAG)** berbasis Knowledge Graph melalui LangChain Core dan LangChain OpenAI
- **Pengambilan Embedding** gejala dari database Neo4j menggunakan Neo4j Python Driver 
- **Indexing & Pencarian Vektor** dengan FAISS 
- **Tokenisasi & n-gram** dengan NLTK 
- **Model Klasifikasi** menggunakan PyTorch
- **Preprocessing & Encoding** teks dengan Transformers dan LabelEncoder dari scikit-learn  
