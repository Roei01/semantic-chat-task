from pathlib import Path
from typing import List, Dict, Iterable, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from models.base import ChatModel, Message

VECTOR_DB_DIR = Path("vectorstore")

class LegalRAGService:
    def __init__(self, chat_model: ChatModel, top_k: int = 100):
        if not VECTOR_DB_DIR.exists():
            raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_DIR}")
            
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name="verdicts"
        )
        
        self.retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        self.chat_model = chat_model
        
    def _is_general_question(self, question: str) -> bool:
        general_keywords = [
            "איזה פסקי", "אילו פסקי", "מה פסק", "כמה פסקי", "רשימה",
            "פסקי דין אתה", "פסק הדין האחרון", "מבחינת התאריך",
            "מה יש במאגר", "מה במאגר", "איזה מסמכים"
        ]
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in general_keywords)
    
    def retrieve(self, question: str) -> List[Document]:
        is_general = self._is_general_question(question)
        
        if is_general:
            query = "פסק דין"
        else:
            query = question
        
        docs = self.retriever.invoke(query)
        
        if not docs:
            return []
        
        stop_words = {
            "של", "את", "על", "כי", "זה", "או", "כל", "הוא", "היא", "גם", "בין", "רק", "אך", "אין", "יש",
            "מה", "מי", "איך", "כיצד", "מתי", "איפה", "למה", "מדוע", "האם",
            "היה", "היתה", "היו", "תהיה", "יהיה",
            "פסק", "דין", "בית", "משפט", "החלטה", "תביעה", "נתבעת", "תובעת", "נגד", "בפני", "ב", "בעמ", 'בע"מ'
        }
        
        raw_words = [w.strip('.,?"\'').lower() for w in question.split()]
        
        filtered_words = []
        for w in raw_words:
            clean_w = w
            if len(w) > 4 and w.startswith(('ב', 'ה', 'ל', 'מ', 'ש', 'כ')):
                clean_w = w[1:]
            
            if len(clean_w) > 1 and clean_w not in stop_words and w not in stop_words:
                filtered_words.append(clean_w)
                
        words = filtered_words
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        
        scored_docs = []
        for idx, doc in enumerate(docs):
            score = (len(docs) - idx) / len(docs) 
            content = doc.page_content
            
            for word in words:
                if word in content: score += 2.0
                if word[::-1] in content: score += 2.0
                if len(word) > 3 and word in content.replace(" ", ""): score += 0.5

            for bg in bigrams:
                if bg in content: score += 5.0
                if bg[::-1] in content: score += 5.0
            
            scored_docs.append((doc, score))
            
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_docs:
            return []
        
        top_score = scored_docs[0][1]
        filtered_docs = []
        for doc, score in scored_docs:
            if score >= top_score * 0.6:
                filtered_docs.append(doc)
        
        return filtered_docs[:3] if filtered_docs else []

    def build_context_and_citations(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        if not docs:
            return "", []
            
        context_parts = []
        citations: List[Dict] = []
        
        for idx, doc in enumerate(docs, start=1):
            cid = f"[{idx}]"
            display_name = doc.metadata.get("display_name", doc.metadata.get("filename", "Unknown"))
            source_path = doc.metadata.get("source_path", "")
            
            file_url = ""
            if source_path:
                from pathlib import Path
                file_path = Path(source_path)
                if file_path.exists():
                    file_url = f"http://localhost:8005/api/files/{file_path.name}"
            
            metadata_str = ""
            if "moddate" in doc.metadata:
                metadata_str += f"\nDate: {doc.metadata.get('moddate', 'Unknown')}"
            if "creationdate" in doc.metadata:
                metadata_str += f"\nCreated: {doc.metadata.get('creationdate', 'Unknown')}"
            
            context_parts.append(
                f"--- DOCUMENT {cid} ---\nSource: {display_name}{metadata_str}\nContent:\n{doc.page_content}\n--- END {cid} ---"
            )
            citations.append({
                "id": cid, 
                "filename": display_name,
                "url": file_url,
                "source_path": source_path,
                "metadata": doc.metadata
            })
            
        return "\n\n".join(context_parts), citations

    def _build_messages(self, question: str, context_text: str, num_sources: int = 0) -> List[Message]:
        if num_sources > 0:
            system_prompt = (
                "אתה עורך דין מומחה בישראל.\n\n"
                "חוקים בלתי משתנים - הפרה של כל אחד מהם היא שגיאה קריטית:\n"
                "1. **אסור לכתוב אף מילה באנגלית!** כל התשובה חייבת להיות בעברית בלבד. אסור לכתוב 'I'll', 'I will', 'answer', 'question', 'directly', 'Hebrew', 'only', 'without', 'writing', 'English', 'repeating' או כל מילה אחרת באנגלית!\n"
                "2. **אסור לכתוב משפטי הקדמה באנגלית!** אל תכתוב 'I'll answer your question directly in Hebrew only' או כל משפט דומה. פשוט ענה ישירות בעברית!\n"
                "3. **אסור לחזור על התשובה!** ענה רק פעם אחת! אם כתבת משפט, אל תכתוב אותו שוב!\n"
                "4. **אסור לכלול את השאלה בתשובה!** אל תכתוב 'שאלה:' או 'תשובה:' - פשוט ענה ישירות!\n"
                "5. התבסס **אך ורק** על המידע המופיע במסמכים המצורפים. אל תמציא מידע.\n"
                "6. **חובה** לציין את המקורות בסוגריים מרובעים [1], [2] וכו' בכל עובדה או קביעה.\n"
                "7. אם המידע לא קיים במסמכים, אמור במפורש: 'לא מצאתי מידע על כך במאגר פסקי הדין'.\n"
                "8. אם הטקסט במסמך משובש או הפוך (בגלל בעיות PDF), נסה להבין את ההקשר כמיטב יכולתך.\n"
                "9. ענה בעברית מקצועית, שוטפת וברורה.\n"
                f"10. סיים את התשובה עם: 'מקורות: [1] שם קובץ ראשון' ואם יש יותר מקורות, המשך: '[2] שם קובץ שני' וכו'.\n\n"
                "**זכור: התחל ישירות בתשובה בעברית. אל תכתוב שום דבר באנגלית. אל תחזור על התשובה.**"
            )
            
            user_content = f"{question}\n\n"
            user_content += f"מקורות מידע מהמאגר ({num_sources} מסמכים):\n{context_text}\n\n"
            user_content += "ענה ישירות בעברית בלבד. התחל ישירות בתשובה - אל תכתוב משפטי הקדמה באנגלית כמו 'I'll answer' או 'I will answer'. אסור לחזור על התשובה. אסור לכלול את השאלה בתשובה."
        else:
            system_prompt = (
                "אתה עורך דין מומחה בישראל.\n\n"
                "חוקים בלתי משתנים:\n"
                "1. **אסור לכתוב אף מילה באנגלית!** כל התשובה חייבת להיות בעברית בלבד - אסור לכתוב 'I'll', 'answer', 'question', 'directly', 'Hebrew' או כל מילה אחרת באנגלית!\n"
                "2. **אסור לחזור על התשובה!** ענה רק פעם אחת!\n"
                "3. **אסור לכלול את השאלה בתשובה!** אל תכתוב 'שאלה:' או 'תשובה:' - פשוט ענה ישירות!\n"
                "4. לא נמצאו מסמכים רלוונטיים במאגר לשאלה זו.\n"
                "5. ענה בעברית מקצועית, שוטפת וברורה.\n"
                "6. **אל תציין מקורות** כי לא נמצאו מסמכים רלוונטיים."
            )
            
            user_content = f"{question}\n\n"
            user_content += "לא נמצאו מסמכים רלוונטיים במאגר לשאלה זו. אמור למשתמש שלא מצאת מידע רלוונטי במאגר פסקי הדין.\n"
            user_content += "ענה ישירות בעברית בלבד. אסור לכתוב באנגלית. אסור לחזור על התשובה. אסור לכלול את השאלה בתשובה."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def _clean_answer(self, answer: str) -> str:
        if not answer:
            return ""
        
        answer = answer.strip()
        
        english_prefixes = [
            "i'll answer",
            "i will answer",
            "i'll answer your question",
            "i will answer your question",
            "let me answer",
            "here's the answer",
            "the answer is",
            "answer:",
            "question:",
            "תשובה:",
            "שאלה:"
        ]
        
        for prefix in english_prefixes:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()
                if answer.startswith(":"):
                    answer = answer[1:].strip()
                break
        
        sentences = answer.split(".")
        filtered_sentences = []
        seen = set()
        
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            
            sent_lower = sent_clean.lower()
            
            if any(phrase in sent_lower for phrase in [
                "i'll answer", "i will answer", "directly in hebrew", 
                "without writing in english", "repeating the question"
            ]):
                continue
            
            if sent_lower not in seen:
                seen.add(sent_lower)
                filtered_sentences.append(sent_clean)
        
        answer = ". ".join(filtered_sentences)

        return answer.strip()
        
    def answer(self, question: str) -> Tuple[str, List[Dict]]:
        docs = self.retrieve(question)
        context, citations = self.build_context_and_citations(docs)
        messages = self._build_messages(question, context, num_sources=len(citations))
        
        answer = self.chat_model.generate(messages)
        answer = self._clean_answer(answer)
        return answer, citations
        
    def stream_answer(self, question: str) -> Tuple[Iterable[Dict[str, str]], List[Dict]]:
        docs = self.retrieve(question)
        context, citations = self.build_context_and_citations(docs)

        messages = self._build_messages(question, context, num_sources=len(citations))
        stream = self.chat_model.stream(messages)

        def cleaned_stream():
            full_text = ""
            for token in stream:
                full_text += token
                yield {"type": "token", "data": token}

            cleaned = self._clean_answer(full_text) or full_text
            yield {"type": "final", "data": cleaned}

        return cleaned_stream(), citations
