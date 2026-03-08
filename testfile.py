# read pypdf and format it normally

from pypdf import PdfReader

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

#OWORKIN ON implementing the above feature in main.py and then test it in testfile.py so that we can read pdf files and extract text from them.  so it is oging to wokr soon. just have to wait and hold ont ight  os it is done