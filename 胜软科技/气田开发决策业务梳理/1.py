import pandas as pd


def excel_to_md(excel_file, sheet_name=0):
    # 读取Excel文件
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

    # 初始化一个空的Markdown内容
    markdown_content = ""

    # 用于追踪每个层级的业务内容
    prev_columns = {
        '业务领域': '',
        '一级业务': '',
        '二级业务': '',
        '三级业务': '',
        '四级业务': '',
        '五级业务': ''
    }

    # 遍历数据并生成Markdown
    for index, row in df.iterrows():
        # 提取业务信息
        business_domain = row['业务领域']
        first_level = row['一级业务']
        second_level = row['二级业务']
        third_level = row['三级业务']
        fourth_level = row['四级业务']
        fifth_level = row['五级业务']
        output_name = row['产出物名称']
        output_style = row['产出物样式']
        processing_rules = row['业务处理'] if not pd.isna(row.get('业务处理', '')) else ''

        # 如果该行的所有业务层级都为空，跳过
        if pd.isna(business_domain) and pd.isna(first_level) and pd.isna(second_level) and pd.isna(third_level) and pd.isna(fourth_level) and pd.isna(fifth_level):
            continue

        # 生成Markdown
        if not pd.isna(business_domain) and business_domain != prev_columns['业务领域']:
            markdown_content += f"# {business_domain}\n"
        if not pd.isna(first_level) and first_level != prev_columns['一级业务']:
            markdown_content += f"## {first_level}\n"
        if not pd.isna(second_level) and second_level != prev_columns['二级业务']:
            markdown_content += f"### {second_level}\n"
        if not pd.isna(third_level) and third_level != prev_columns['三级业务']:
            markdown_content += f"#### {third_level}\n"
        if not pd.isna(fourth_level) and fourth_level != prev_columns['四级业务']:
            markdown_content += f"##### {fourth_level}\n"
        if not pd.isna(fifth_level) and fifth_level != prev_columns['五级业务']:
            markdown_content += f"###### {fifth_level}\n"

        # 添加产出物及相关内容
        if not pd.isna(output_name):
            markdown_content += f"- {output_name}\n"
            if not pd.isna(output_style):
                markdown_content += f"  - 产出物样式: {output_style}\n"
            if processing_rules:
                markdown_content += "  - 处理规则:\n"
                for rule in processing_rules.split('\n'):
                    markdown_content += f"    - {rule.strip()}\n"

        # 更新上一级的业务内容
        prev_columns['业务领域'] = business_domain
        prev_columns['一级业务'] = first_level
        prev_columns['二级业务'] = second_level
        prev_columns['三级业务'] = third_level
        prev_columns['四级业务'] = fourth_level
        prev_columns['五级业务'] = fifth_level

    return markdown_content


def save_to_md(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    excel_file = '安全业务体系梳理 - 1209.xlsx'  # Excel文件路径
    markdown_file = 'output.md'  # 输出的Markdown文件路径

    md_content = excel_to_md(excel_file)
    save_to_md(markdown_file, md_content)
    print(f"Markdown文件已保存到 {markdown_file}")
