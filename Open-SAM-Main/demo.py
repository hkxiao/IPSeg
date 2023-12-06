names = ['Alice', 'Bob', 'Charlie']
scores = [85, 90, 75]

for index, (name, score) in enumerate(zip(names, scores)):
    index = 10  # 这会导致后续循环中的错误
    print(f'{index + 1}. {name}: {score}')