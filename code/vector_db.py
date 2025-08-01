from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings


pdf_data_path = "data"
vector_db_path = "vector_db/db_faiss"

def create_db_from_text():
    raw_docs = '''Chia sẻ với Tuổi Trẻ, nhiều chuyên gia truyền thông nhận định việc những người có ảnh hưởng tiếp tay quảng cáo sữa giả cũng gây tác hại không kém những người sản xuất, bởi đông đảo người tiêu dùng mua sản phẩm vì tin vào hình ảnh uy tín của người quảng cáo mời gọi.

Bà Tô Giang, giám đốc Công ty tư vấn Renaissance, cho biết sữa là dinh dưỡng thiết yếu cho trẻ em, bà bầu, người già, bệnh nhân.

Sữa giả (chất lượng dưới 70% công bố) gây nguy cơ suy dinh dưỡng, ảnh hưởng thai nhi hoặc làm nặng bệnh mạn tính. "Với quy mô 500 tỉ đồng, vụ việc làm xói mòn niềm tin vào thị trường sữa, khiến người tiêu dùng hoang mang, doanh nghiệp chân chính mất khách.

Là mẹ, tôi lo khi chọn sữa cho con và cha mẹ, sợ hàng giả len lỏi và cũng mất niềm tin vì không biết nguồn hàng nào mới là thật", bà Giang cho biết.

Ông Nguyễn Duy Vĩ, giám đốc Công ty truyền thông Buzi, cho rằng trong khi vụ "kẹo rau" từng khiến cộng đồng xôn xao vì chiêu trò thổi phồng công dụng, vụ sữa giả lần này lại là hồi chuông cảnh báo cho toàn xã hội và các cơ quan quản lý nhà nước. "Không còn là chiêu trò, đây là một hành vi lừa đảo có tổ chức, có mục đích và bất chấp hậu quả để trục lợi", ông Vĩ nhận xét.
'''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(raw_docs)

    embedding = GPT4AllEmbeddings(model = "models/all-MiniLM-L6-v2-fp16.gguf")
    
    db = FAISS.from_texts(chunks, embedding)
    db.save_local(vector_db_path)
    print(f"Vector database created and saved at {vector_db_path}")
    return db
create_db_from_text()
def create_db_from_pdf():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = text_splitter.split_documents(docs)

    embedding = GPT4AllEmbeddings(model= "models/all-MiniLM-L6-v2-fp16.gguf")
    
    db = FAISS.from_documents(split_docs, embedding)
    db.save_local(vector_db_path)
    print(f"Vector database created and saved at {vector_db_path}")
    return db   
create_db_from_pdf()