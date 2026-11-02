import efinance as ef

# 一行代码获取全市场实时行情DataFrame
df = ef.stock.get_realtime_quotes()
print(df.head())
