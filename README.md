# kaggle-titanic-79.665%  
  
  
  
## Data pre-processing
- Mapping 18 First names into 6(Royal, Special, Mr, Mrs, Miss, Master)  
- Used just 5 features(Age, Fare, Sex, Pclass, Title)  
- Filling null data  
1. Age  
```python
age_analysis = titanic.groupby(["Sex", "Title"])
age_analysis.Age.median() titanic.Age 
age_analysis.Age.apply(lambda x: x.fillna(x.median()))
```
2. Fare
```python
titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())
titanic.Fare = titanic.Fare.map(lambda i: np.log(i) if i > 0 else 0)
```

## Prediction model
```python
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 5)
AC = []
for train, validation in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(20, input_dim=12, activation="relu"))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X[train], Y[train], epochs=110, batch_size=5)
    k_accuracy = '%.4f' % (model.evaluate(X[validation], Y[validation])[1])
    AC.append(k_accuracy)
```

