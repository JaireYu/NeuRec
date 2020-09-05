pos_items = {}
neg_items = {}
with open('./Yahootrain') as f:
    for line in f.readlines():
        user = int(line.split('::::')[0])
        item = int(line.split('::::')[1])
        if user not in pos_items.keys():
            pos_items[user] = []
        else:
            pos_items[user].append(item)
with open('./Yahootest') as f:
    for line in f.readlines():
        user = int(line.split('::::')[0])
        item = int(line.split('::::')[1])
        pos_items[user].append(item)
with open('./Yahoodev') as f:
    for line in f.readlines():
        user = int(line.split('::::')[0])
        item = int(line.split('::::')[1])
        pos_items[user].append(item)
with open('./Yahooexpose_train') as f:
    for line in f.readlines():
        user = int(line.split('::::')[0])
        neg_items[user] = []
        cand_item = line.split('::::')[1]
        cand_items = cand_item.split()
        for i in cand_items:
            if int(i) not in pos_items[user]:
                neg_items[user].append(str(i))
with open('./Yahoo.expose', 'w') as f:
    user_list = [key for key in neg_items.keys()]
    user_list = sorted(user_list)
    for user in user_list:
        f.write(':'.join([str(user)]+neg_items[user])+'\n')
