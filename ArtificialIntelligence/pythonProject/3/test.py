# one-hot编码
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()
data1=np.array([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
# print(data1)
# encoder.fit(data1)
# encode_data=encoder.transform(data1)
encode_data=encoder.fit_transform(data1)
# print(encode_data)
print(encode_data.toarray())
test=np.array([[2,3,5,3]])
# encoder_vector=encoder.transform(test).toarray()
# print(encoder_vector)