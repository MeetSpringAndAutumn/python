import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import pymysql
import re


# 1. 读取Excel文件
def read_excel(file_path, sheet_name, columns):
    # 读取Excel文件的指定工作表和列
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)
    return df


# 2. 连接数据库
def create_db_connection():
    # 使用SQLAlchemy连接MySQL数据库
    engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/scyxzngk')
    return engine


# 3. 查询数据库中的现有数据
def fetch_existing_data(connection, jhdm):
    # 根据JHDM从数据库中查询现有的数据
    query = "SELECT SJDWMC, SJDWDM, JGMCLJ, JGDMLJ FROM SCYX_JCPZ_JXX WHERE JHDM = %s"
    result = connection.execute(query, (jhdm,))
    row = result.fetchone()
    return row if row else None


# 4. 更新数据库表
def update_database(df, engine):
    with engine.connect() as connection:
        for index, row in df.iterrows():
            jhdm = row['JHDM']
            sjdwmc = row['SJDWMC']
            jgmclj = row['JGMCLJ']
            jgdmlj = row['JGDMLJ']

            # 解析JGDMLJ，提取最后的SJDWDM（即JHXXXXX格式的数字）
            sjdwdm = jgdmlj.split('/')[-2]

            # 获取当前时间
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 查询数据库中现有的数据
            existing_data = fetch_existing_data(connection, jhdm)

            # 如果现有数据与新数据不同，则进行更新
            if existing_data:
                # 对比现有数据和新数据
                existing_sjdwmc, existing_sjdwdm, existing_jgmclj, existing_jgdmlj = existing_data
                if (sjdwmc != existing_sjdwmc or
                        sjdwdm != existing_sjdwdm or
                        jgmclj != existing_jgmclj or
                        jgdmlj != existing_jgdmlj):

                    # 编写SQL语句，使用参数化查询来防止SQL注入
                    sql = """
                    UPDATE SCYX_JCPZ_JXX
                    SET SJDWMC = %s, SJDWDM = %s, JGMCLJ = %s, JGDMLJ = %s, XGSJ = %s
                    WHERE JHDM = %s
                    """
                    # 执行SQL语句
                    connection.execute(sql, (sjdwmc, sjdwdm, jgmclj, jgdmlj, current_time, jhdm))
                    print(
                        f"Updated JHDM: {jhdm} with SJDWMC: {sjdwmc}, SJDWDM: {sjdwdm}, JGMCLJ: {jgmclj}, JGDMLJ: {jgdmlj}, XGSJ: {current_time}")
                else:
                    print(f"No changes for JHDM: {jhdm}. Data is the same.")
            else:
                print(f"No record found for JHDM: {jhdm}, skipping update.")


# 5. 主函数
def main():
    # 读取Excel文件
    group_file = '班组信息.xlsx'  # 班组信息.xlsx文件路径
    well_file = '生产运行智能管控数据治理模板V2.0（基础填写）--清河 (1).xlsx'  # 井基础信息表.xlsx文件路径

    # 读取班组信息.xlsx文件
    group_df = read_excel(group_file, sheet_name='Sheet1', columns=[2, 3, 4])
    group_df.columns = ['SJDWMC', 'JGMCLJ', 'JGDMLJ']  # 给列命名

    # 读取井基础信息表.xlsx文件
    well_df = read_excel(well_file, sheet_name='井基础信息表', columns=[1, 5])
    well_df.columns = ['JHDM', 'SJDWMC']  # 给列命名

    # 合并两个DataFrame，根据SJDWMC进行合并
    merged_df = pd.merge(well_df, group_df, on='SJDWMC', how='left')

    # 连接数据库
    engine = create_db_connection()

    # 更新数据库
    update_database(merged_df, engine)
    print("Database update completed.")


if __name__ == '__main__':
    main()
