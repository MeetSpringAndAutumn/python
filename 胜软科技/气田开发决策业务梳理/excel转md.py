import shutil
import zipfile
import os
import xml.etree.ElementTree as ET
import openpyxl
import pandas as pd

image_list = []  # 存放从excel读出的DISPIMG_id
image_dict = {}  # 存放图片ID和文件路径的映射

def read_excel_data(filename_path):
    """读取Excel数据并提取图片ID"""
    workbook = openpyxl.load_workbook(filename_path, data_only=False)
    sheet = workbook.active

    data = []
    for row in sheet.iter_rows(min_row=1, values_only=False):
        row_data = []
        for cell in row:
            if cell.value and isinstance(cell.value, str) and '=_xlfn.DISPIMG(' in cell.value:
                formula = cell.value
                start = formula.find('"') + 1
                end = formula.find('"', start)
                image_id = formula[start:end]
                row_data.append(image_id)
                image_list.append(image_id)
            else:
                row_data.append(cell.value)
        data.append(row_data)
    return data

def get_xml_id_image_map(xlsx_file_path):
    """获取XML中图片ID和路径的映射"""
    with zipfile.ZipFile(xlsx_file_path, 'r') as zfile:
        with zfile.open('xl/cellimages.xml') as file:
            xml_content = file.read()
        with zfile.open('xl/_rels/cellimages.xml.rels') as file:
            relxml_content = file.read()

    root = ET.fromstring(xml_content)
    namespaces = {
        'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
    }

    name_to_embed_map = {}
    for pic in root.findall('.//xdr:pic', namespaces=namespaces):
        name = pic.find('.//xdr:cNvPr', namespaces=namespaces).attrib['name']
        embed = pic.find('.//a:blip', namespaces=namespaces).attrib[
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed']
        name_to_embed_map[name] = embed

    root1 = ET.fromstring(relxml_content)
    namespaces = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}

    id_target_map = {child.attrib['Id']: child.attrib.get('Target', 'No Target Found')
                     for child in root1.findall('.//r:Relationship', namespaces=namespaces)}

    name_to_target_map = {name: id_target_map[embed]
                         for name, embed in name_to_embed_map.items()
                         if embed in id_target_map}
    return name_to_target_map

def create_image_dict(xlsx_file_path, output_folder='images'):
    """创建图片ID和文件路径的映射字典"""
    read_excel_data(xlsx_file_path)
    name_to_target_map = get_xml_id_image_map(xlsx_file_path)
    new_map = {key: name_to_target_map.get(key) for key in image_list if key in name_to_target_map}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with zipfile.ZipFile(xlsx_file_path, 'r') as zfile:
        for key, image_path in new_map.items():
            actual_image_path = f'xl/{image_path}'
            if actual_image_path in zfile.namelist():
                new_file_path = os.path.join(output_folder, f"{key}.png")
                with zfile.open(actual_image_path) as image_file:
                    image_content = image_file.read()
                    with open(new_file_path, 'wb') as new_file:
                        new_file.write(image_content)
                image_dict[key] = f"{output_folder}/{key}.png"

def clean_images_folder(folder_path):
    """清空图片文件夹"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def save_to_md(file_path, content):
    """保存内容到Markdown文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def excel_to_md(excel_file, sheet_name=0, output_folder='images'):
    """将Excel转换为Markdown"""
    create_image_dict(excel_file, output_folder)
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
    markdown_content = ""

    prev_columns = {
        '业务场景': '',
        '一级节点': '',
        '二级节点': '',
        '三级节点': '',
        '末级工作节点': '',
        '五级业务': ''
    }

    for index, row in df.iterrows():
        # 提取各列数据
        business_domain = str(row['业务场景']) if not pd.isna(row['业务场景']) else ''
        first_level = str(row['一级节点']) if not pd.isna(row['一级节点']) else ''
        second_level = str(row['二级节点']) if not pd.isna(row['二级节点']) else ''
        third_level = str(row['三级节点']) if not pd.isna(row['三级节点']) else ''
        fourth_level = str(row['末级工作节点']) if not pd.isna(row['末级工作节点']) else ''
        fifth_level = str(row['五级业务']) if not pd.isna(row['五级业务']) else ''
        output_name = str(row['产出物名称']) if not pd.isna(row['产出物名称']) else ''
        output_style = str(row['产出物样式']) if not pd.isna(row['产出物样式']) else ''
        processing_rules = str(row['业务处理']) if not pd.isna(row.get('业务处理', '')) else ''

        if not any([business_domain, first_level, second_level, third_level, fourth_level, fifth_level]):
            continue

        # 生成各级标题
        if business_domain and business_domain != prev_columns['业务场景']:
            markdown_content += f"# {business_domain}\n"
        if first_level and first_level != prev_columns['一级节点']:
            markdown_content += f"## {first_level}\n"
        if second_level and second_level != prev_columns['二级节点']:
            markdown_content += f"### {second_level}\n"
        if third_level and third_level != prev_columns['三级节点']:
            markdown_content += f"#### {third_level}\n"
        if fourth_level and fourth_level != prev_columns['末级工作节点']:
            markdown_content += f"##### {fourth_level}\n"
        if fifth_level and fifth_level != prev_columns['五级业务']:
            markdown_content += f"###### {fifth_level}\n"

        # 添加产出物及相关内容
        if output_name:
            markdown_content += f"- {output_name}\n"
            # 在处理产出物样式时，检查并转换DISPIMG格式
            if output_style:
                markdown_content += "  - 产出物样式\n"
                if '=DISPIMG' in output_style:
                    # 提取图片ID
                    img_id = output_style[output_style.find('"') + 1:output_style.rfind('"')]
                    if img_id in image_dict:
                        # 只显示图片，不带文字说明
                        markdown_content += f"    - ![](images/{img_id}.png)\n"
                    else:
                        markdown_content += f"    - {output_style}\n"
                else:
                    markdown_content += f"    - {output_style}\n"
            if processing_rules and processing_rules != 'nan':
                markdown_content += "  - 处理规则\n"
                for rule in processing_rules.split('\n'):
                    if rule.strip():
                        markdown_content += f"    - {rule.strip()}\n"

        # 更新前一行的值
        prev_columns.update({
            '业务场景': business_domain,
            '一级节点': first_level,
            '二级节点': second_level,
            '三级节点': third_level,
            '末级工作节点': fourth_level,
            '五级业务': fifth_level
        })

    return markdown_content

if __name__ == '__main__':
    excel_file = '安全环保IPOM表-20250224-生董董.xlsx'
    markdown_file = 'output.md'
    output_folder = 'images'

    clean_images_folder(output_folder)
    md_content = excel_to_md(excel_file, output_folder=output_folder)
    save_to_md(markdown_file, md_content)
    print(f"Markdown文件已保存到 {markdown_file}")