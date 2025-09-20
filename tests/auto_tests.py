import inspect
import def_tests as auto_tests

def run_all_tests():
    print("="*60)
    print("Запуск всех тестов...")
    print("="*60)

    total = 0
    passed = 0
    failed = 0

    #Получаем все функции из def_tests, начинающиеся с 'test_'
    test_functions = [
        func for name, func in inspect.getmembers(auto_tests, inspect.isfunction)
        if name.startswith('test_')
    ]

    for test_func in test_functions:
        total += 1
        print(f"\nЗапуск: {test_func.__name__}... ", end="")

        try:
            result = test_func()
            if result is True:
                print("УСПЕШНО")
                passed += 1
            else:
                print("ПРОВАЛ (вернул False)")
                failed += 1
        except Exception as e:
            print(f"ОШИБКА: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТ: {passed}/{total} пройдено, {failed} упало")
    print("="*60)

    if failed > 0:
        exit(1) 
    else:
        exit(0)

if __name__ == "__main__":
    run_all_tests()