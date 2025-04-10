import pandas as pd
import pymysql
from sqlalchemy import create_engine

def main():
    # 配置数据库连接
    db_config = {
        'host': '10.85.2.9',        # 数据库地址
        'port': 33090,              # 数据库端口
        'user': 'root',             # 数据库用户名
        'password': '123456',       # 数据库密码
        'database': 'scyxzngk',     # 数据库名称
        'charset': 'utf8mb4'        # 字符编码
    }

    # 创建数据库连接
    engine = create_engine('mysql+pymysql://root:YTHT2022@10.85.2.9:33090/scyxzngk')
    connection = engine.connect()

    # 读取 Excel 文件
    excel_file = '生产运行智能管控数据治理模板V2.0（基础填写)--涪陵页岩气公司（1.16） (1).xlsx'  # 替换为你的 Excel 文件名
    sheet_name = '井基础信息表'            # 替换为你的工作表名
    data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # 开启事务
    trans = connection.begin()
    try:
        for _, row in data.iterrows():
            if pd.isna(row['标准井号']) or pd.isna(row['汉字井号']) or pd.isna(row['井别']) or pd.isna(
                    row['巡检班组(需核对)']):
                print(
                    f"关键字段为空，跳过该行数据: 标准井号={row['标准井号']}, 汉字井号={row['汉字井号']}, 井别={row['井别']}, 巡检班组={row['巡检班组(需核对)']}")
                continue  # 跳过当前循环
            标准井号 = row['标准井号']
            汉字井号 = row['汉字井号']
            井别 = row['井别'] or '未知井别'
            巡检班组 = row['巡检班组(需核对)'] or '未知班组'
            机构路径名称 = '清河采油厂'


            print(标准井号)
            # 构造插入语句
            sql = f"""
                INSERT INTO SCYX_JCPZ_JXX
                SELECT
                    UUID(), NOW(), CZRIP, CZRXM, CZRBS, CZBM, CZBMBS, NOW(), XGRIP, XGRXM, XGRBS,
                    ZKMC, ZKDM, JCDM, LXZKJCDM,  '{汉字井号}', '{标准井号}',CTJ, JHLX, JHLXBS,
                    SYZTMC, SYZTBS, SJDWMC, SJDWDM, JGMCLJ, JGDMLJ, GLRMC, GLRBS, SSEJDWMC, SSEJDWDM, PX,
                    SFQY, SFSC, SCSJ, SCRIP, SCRXM, SCRBS, BEIZHU, MIAOSHU, CJZBRYID, CJZBRYXM, KSCJSJ,
                    JSCJSJ, IFCJ, GLQDM, GLQMC, JBDM, JBMC, JXDM, JXMC, JHDMEPBP
                FROM SCYX_JCPZ_JXX
                WHERE SSEJDWMC = '{机构路径名称}' AND SJDWMC = '{巡检班组}' AND JHLX = '{井别}'
                LIMIT 1;
                """
            # 执行 SQL
            connection.execute(sql)

        # 提交事务
        trans.commit()
        print("所有数据插入成功！")

    except Exception as e:
        # 回滚事务
        trans.rollback()
        print(f"插入数据失败，已回滚事务。错误信息: {e}")

    finally:
        # 关闭连接
        connection.close()
        print("数据库连接已关闭。")

if __name__ == "__main__":
    main()

# # 遍历 Excel 数据并生成和执行 SQL 语句
# for _, row in data.iterrows():
#     标准井号 = row['标准井号']
#     汉字井号 = row['汉字井号']
#     井别 = row['井别']
#     巡检班组 = row['巡检班组(需核对)']
#     机构路径名称 = row['机构路径名称（需核对）']
#
#     # 构造插入语句
#     sql = f"""
#     INSERT INTO SCYX_JCPZ_JXX
#     SELECT
#         UUID(), NOW(), CZRIP, CZRXM, CZRBS, CZBM, CZBMBS, NOW(), XGRIP, XGRXM, XGRBS,
#         ZKMC, ZKDM, JCDM, LXZKJCDM, '{标准井号}', '{汉字井号}', CTJ, JHLX, JHLXBS,
#         SYZTMC, SYZTBS, SJDWMC, SJDWDM, JGMCLJ, JGDMLJ, GLRMC, GLRBS, SSEJDWMC, SSEJDWDM, PX,
#         SFQY, SFSC, SCSJ, SCRIP, SCRXM, SCRBS, BEIZHU, MIAOSHU, CJZBRYID, CJZBRYXM, KSCJSJ,
#         JSCJSJ, IFCJ, GLQDM, GLQMC, JBDM, JBMC, JXDM, JXMC, JHDMEPBP
#     FROM SCYX_JCPZ_JXX
#     WHERE SSEJDWMC = '{机构路径名称}' AND SJDWMC = '{巡检班组}' AND JHLX = '{井别}'
#     LIMIT 1;
#     """
#
#     # 执行 SQL
#     try:
#         connection.execute(sql)
#         print(f"成功插入数据: {标准井号}")
#     except Exception as e:
#         print(f"插入数据失败: {标准井号}, 错误信息: {e}")
#
# # 关闭连接
# connection.close()
# print("所有数据处理完成，数据库连接已关闭。")
