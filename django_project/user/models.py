from django.db import models
from django.contrib.auth.models import User
from upload.models import *

from django.core.files.storage import FileSystemStorage


def create_file_path(instance, filename):
    return 'user_{0}/file/{1}'.format(instance.user.id, filename)


def create(user_id, name):
    return 'user_{0}/file/{1}'.format(user_id, name)

# def check(path,type):
#     with open(path, 'rb') as f:
#
#
#         return True


# Create your models here.
class UserFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 用户
    type = models.BooleanField(default=True)  # True为训练
    train_id = models.ForeignKey(Train, on_delete=models.CASCADE, null=True)  # 训练的id
    test_id = models.ForeignKey(Test, on_delete=models.CASCADE, null=True)  # 测试的id
    path = models.CharField(max_length=100)  # 文件的上传路径

    # ？？path怎么存，还是用filefield存，这里不会，导致不好处理用户上传的训练和测试的逻辑函数，用来训练和测试的算法我们都有
    flag = models.BooleanField(default=True)  # 文件检验通过
    # merge = models.BooleanField(default=False)  # 融合文件标志True

    def __init__(self, **kwargs):
        super(UserFile, self).__init__(**kwargs)
        self.user = kwargs.get('user', '')
        self.type = kwargs.get('type', '')
        # if self.type:
        #     self.train_id = kwargs.get('id', '')
        # else:
        #     self.test_id = kwargs.get('id', '')
        # self.path = kwargs.get('path', '')
        # self.flag = check(self.file.path, self.type)
        # self.file.


