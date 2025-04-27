Docelowo chcemy zrobic sobie DataFrame w stylu:

**topic_id | topic_name | topic_overview | document_id | document_text | label**



Najlatwiej (chyba) bedzie to zrobic przez połączenie dwóch DataFrame'ów:

###### 1. Topics

W folderze data/DATA2019 są podfoldery m.in. DTA.

a) jesli bierzemy sam opis to z folderu topics bierzemy kolumny: **topic_id, topic_name, topic_overview**
b) jesli chcemy wiecej kolumn to grzebiemy w folderze protocols

###### 2. Documents

Zeby dostac dane dot. dokumentów

* [ ] Pobieramy pliki xml (bez md5) ze strony: [https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
* [ ] Z kazdego wyciagamy jego PMID (ew. inne ID ale to jest chyba dobre) (**document_id**) oraz **document_text** (nie znalazlem na razie jakie to pole) + tytul czy inne co uznamy za potrzebne

###### Połączenie

Na koniec w folderze data/train i data/test dla kazdego tematu mamy trzy pliki. Interesuje nas glownie **.pids**
Na bazie tych plików matchujemy, odpowiednie wiersze z 1. Topics z odpowiednikami z 2. Documents. 

*One wszystkie jak rozumiem mają label 1, więc musimy sztucznie pomieszać i dołożyć jakieś dane, żeby mieć też dane na 0
