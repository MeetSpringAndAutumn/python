def find_combinations(target, min_coin_index, coins, current_combination, combinations):
    if target == 0:
        combinations.append(current_combination)
        return

    for i in range(min_coin_index, len(coins)):
        coin = coins[i]
        if target >= coin:
            find_combinations(target - coin, i, coins, current_combination + [coin], combinations)


def main():
    target = 1314
    coins = [100, 50, 20, 10, 5, 1]
    combinations = []
    find_combinations(target, 0, coins, [], combinations)

    print("所有可能的组合如下：")
    for combination in combinations:
        if len(combination) <= 10:
            print(combination)


if __name__ == "__main__":
    main()
