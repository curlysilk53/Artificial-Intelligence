import pylab as plt
import string
import re

f = open('wiki.txt', 'r')
data_text = f.readlines() # считывание из файла
data_text_word = []  # словарь всех слов

for line in data_text: # считывание всех строк
    line = line.lower() # приведение к нижнему регистру
    line = re.sub('unk', '', line) # удалить слово unk
    line = "".join(i for i in line if i not in string.punctuation)# удаление всех знаков препинания
    result = "".join([i for i in line if not i.isdigit()]) # удаление всез цифр
    line = line.split() # разбить строку на части
    data_text_word.extend(line) # добавть строку
    
frequency = {} # cчитаем частоту слов
for word in data_text_word:
    count = frequency.get(word,0)
    frequency[word] = count + 1
     
frequency_list = frequency.keys() # представление ключей словаря
list = [] 
for words in frequency_list:
    list.append([words, frequency[words]])

list.sort(key=lambda x: x[1]) # сортировать по частоте

s=[]
n=[]
for i in reversed(list[-10:]):
    s.append(i[1]) # слово
    n.append(i[0]) # частота
x=range(len(s)) # количество столбцов
ax = plt.gcf()
ax.canvas.set_window_title('Самые распростанённые слова wiki.txt')
ax = plt.gca()
ax.bar(x, s, align='edge')
ax.set_xticklabels(n) # написать слово 
ax.set_xticks(x) # cлово под столбцом
plt.show()