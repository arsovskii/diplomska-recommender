# Интерактивен систем за препорака
Систем што интегрира GNN-базиран backend со интерактивна веб страна за предлози за книги во реално време

---

## Вклучува:
- **Graph Neural Networks:** Имплементација на GraphSAGE, GAT, Transformer базирани модели за анализирање на врски помеѓу корисници и книги
- **Интерактивен Frontend:** Направено со Sveltekit за динамична интеракција на корисникот
- **RESTful API:** Користејќи Flask за поврзување на моделот со корисниците
- **Dataset со книги:** Превземено од Kaggle, збогатено со јавни податоци од Google и Wikipedia, прочистено и претпроцесирано за најдобри резултати при тренирање

---

## Requirements

### Backend Requirements
- Python 3.8+
- Libraries:
  - PyTorch 
  - PyTorch Geometric
  - Flask
  - pandas
  - numpy
  - scikit-learn

### Frontend Requirements
- Node.js
- Libraries:
  - SvelteKit
  - DaisyUI
  - TailwindCSS

---
##Screenshot
![image](https://github.com/user-attachments/assets/4e92702e-abfd-4bb4-a925-154cd4071177)


## Инструкции:
```bash
git clone https://github.com/arsovskii/diplomska-recommender.git
cd book-recommendation-system
```
# 1. Backend
Инсталирање на барања

```bash
pip install -r requirements.txt
```

```bash
cd book-recommender-backend
```

Стартување на сервер
```bash
flask run
ИЛИ
python app.py
```

# 2.  Frontend
```bash
cd book-recommender-frontend
```
Инсталирање на барања
```bash
npm install
```
Стартување на Sveltekit
```bash
npm run dev
```
