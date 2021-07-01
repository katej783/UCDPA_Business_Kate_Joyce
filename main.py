import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re

book = pd.read_csv('books.csv', error_bad_lines=False)

print(book.isnull().sum())
print(book.dtypes)

books = book.drop(columns=['bookID', 'isbn', 'isbn13'], axis=1)

print(books.head())
print(books.shape)

missing_values_count = books.isnull().sum()
print(missing_values_count)

drop_duplicates = books.drop_duplicates()
print(books.shape, drop_duplicates.shape)

books.replace(to_replace='J.K. Rowling/Mary GrandPré', value='J.K. Rowling', inplace=True)

total_cells = np.product(books.shape)
total_missing = (missing_values_count.sum())
percent_missing = (total_missing/total_cells)
print(percent_missing)

authors_dict = books['authors'].value_counts().to_dict()
print(authors_dict)

total_authors_dict = 0
for value in authors_dict.values():
    total_authors_dict += value
print(total_authors_dict)

# OR

total_authors_dict1 = sum([value for value in authors_dict.values()])
print(total_authors_dict1)

most_common_authors = books['authors'].value_counts()[:40]
print(most_common_authors)

print(most_common_authors['John Grisham'])

top_books = books[books['ratings_count'] > 100000]
top_20 = top_books.sort_values('average_rating', ascending=False).head(20)
print(top_20)

most_rated = top_books.sort_values('ratings_count', ascending=False).head(20)
print(most_rated)

top_5 = top_20.iloc[[0, 1, 2, 3, 4], [0, 1, 2]]
print(top_5)

language = books.groupby('language_code')['authors'].count()
print(language)

sns.barplot(x=most_common_authors, y=most_common_authors.index)
plt.subplots_adjust(left=0.3)
plt.title("40 Authors with Most Books")
plt.subplots_adjust(top=0.9)
plt.show()

fig, ax = plt.subplots()
x = books['title'].head(5)
y = books['num_pages'].head(5)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title("Pages")
plt.subplots_adjust(top=0.9)
ax.bar(x, y)
plt.show()

fig1, ax = plt.subplots()
x2 = top_20['title'].head(10)
y2 = books['average_rating'].head(10)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('Top Rated Books')
plt.subplots_adjust(top=0.9)
ax.bar(x2, y2)
plt.show()

# Regex
txt = "harry potter is half wizard"
x = re.search("^harry.*wizard$",txt)
if x:
    print("Match!")
else:
    print("No Match")

x = re.findall("ha", txt)
print(x)

x = re.split("\s", txt)
print(x)

X_train, X_test, y_train, y_test = train_test_split(books['num_pages', 'ratings_count', 'text_reviews_count'], books['average_rating'], test_size=0.2, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)
svc.predict(X_test)
svc.score(X_test, y_test)

