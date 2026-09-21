from sqlalchemy.orm import Session
from app.database.models import MemoryEntry

def retrieve_memory(db:Session,query:str|None=None,limit:int=20):
    items=db.query(MemoryEntry).order_by(MemoryEntry.updated_at.desc()).limit(100).all()
    if query:
        terms=set(query.lower().split()); items=sorted(items,key=lambda m:len(terms & set((m.key+" "+str(m.value)).lower().split())),reverse=True)
    return [{"id":m.id,"memory_type":m.memory_type,"key":m.key,"value":m.value,"confidence":m.confidence,"session_id":m.session_id,"updated_at":m.updated_at.isoformat()} for m in items[:limit]]

def save_memory(db:Session,memory_type:str,key:str,value:dict,confidence:float=1.0,session_id=None):
    found=db.query(MemoryEntry).filter_by(memory_type=memory_type,key=key).first()
    if found: found.value=value; found.confidence=confidence; found.session_id=session_id
    else: found=MemoryEntry(memory_type=memory_type,key=key,value=value,confidence=confidence,session_id=session_id); db.add(found)
    db.commit(); db.refresh(found); return found

def forget_memory(db:Session,memory_id:str):
    item=db.get(MemoryEntry,memory_id)
    if not item: return False
    db.delete(item); db.commit(); return True

def clear_memory(db:Session):
    count=db.query(MemoryEntry).delete(); db.commit(); return count
