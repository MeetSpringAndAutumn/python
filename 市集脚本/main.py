from time import sleep

import requests
import json

url = "https://mall.bilibili.com/mall-magic-c/internet/c2c/v2/list"

i_want = []
nextId = None
while True:
    payload = json.dumps({
        "categoryFilter": "2273",
        "priceFilters": [
            "5000-900000"
        ],
        "discountFilters": [
            "10-100"
        ],
        "nextId": nextId
    })

    headers = {
        'authority': 'mall.bilibili.com',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5,ja;q=0.4',
        'content-type': 'application/json',
        'cookie': 'buvid4=F2F80493-49E2-C3BE-6765-58A7506694C873965-022022519-uk4/G28K78wsCqcup9Yd8FeAZDPimJoXGcUdVNb8P3UL7QrncOSDNA%3D%3D; LIVE_BUVID=AUTO9416463988500442; DedeUserID=1884657316; DedeUserID__ckMd5=ffa3209dc2d3482a; PVID=1; CURRENT_FNVAL=4048; home_feed_column=5; header_theme_version=CLOSE; buvid_fp=b3e76b039cad5c8b2abcfd643d657930; rpdid=|(u)mJ~~|u~)0J\'u~|mkukuuk; buvid3=6297A5D9-331F-4A2A-3CF4-0008F4AD8AFB87171infoc; b_nut=1716728887; _uuid=263FB51A-963C-8FA10-B125-D69C963A3381088147infoc; b_lsid=8BB6B1A2_1904FF74A10; bsource=search_bing; enable_web_push=DISABLE; browser_resolution=1482-792; SESSDATA=d906c001%2C1734880530%2Cdb687%2A61CjDT6GkzvNqHCFs4Na9Ay4MaSsgK3spKQ9mTE-1Vy4TXtPLwVY_uUNNJQBZX0yEWi44SVm50WlpCdHF1Mlk5R0pQSmtrRmVmOXN3c0pUU3hYMWk5MTZQb0llb0Q4aDhQZEJ1M1N5UloyUTRiUEpHOVdJZWJDZno4dmVTbTZEdHBYa3FjNHcyM3pnIIEC; bili_jct=ff3ac33ea583457fa771532067b7a874; sid=5xfr5f63; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTk1ODc3MzYsImlhdCI6MTcxOTMyODQ3NiwicGx0IjotMX0.n2AfiPYB96GbINkzxSIwyvx6TKXNXLYg5zkcSDv9xsU; bili_ticket_expires=1719587676',
        'origin': 'https://mall.bilibili.com',
        'referer': 'https://mall.bilibili.com/neul-next/index.html?page=magic-market_index',
        'sec-ch-ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        # print(response.text)
        response = response.json()
        nextId = response["data"]["nextId"]
        if nextId is None:
            break
        data = response["data"]["data"]
        for item in data:
            name = item["c2cItemsName"]
            if "无尽夏" in name:
                if item not in i_want:
                    i_want.append(item)
                    # detail = item['detailDtoList'][0]  # 访问列表中的第一个元素
                    # print(detail)
                    # detail_name = detail.get('blindBoxId', 'N/A')
                    # show_price = detail.get('showPrice', 'N/A')
                    # show_market_price = detail.get('showMarketPrice', 'N/A')
                    # print(detail_name)
                    # print(item)
                    print(f"{item['c2cItemsName']},{item['showPrice']},{item['showMarketPrice']},{item['c2cItemsId']}")
                    print()
                    # print(item['detailDtoList']['name'],item['detailDtoList']['showPrice'],item['detailDtoList']['showMarketPrice'])
        sleep(1)
    except Exception as e:
        sleep(3)

print(i_want)

min_element = min(i_want, key=lambda x: x["price"])
for item in i_want:
    print(f"{item['c2cItemsName']},{item['c2cItemsId']},{item['price']}")
print(min_element)
