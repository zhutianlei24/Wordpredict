import tflearn
from numpy import *
count = 0
class Code_Completion_Baseline:

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number) #initial a vector with how many tstrings
        vector[self.string_to_number[string]] = 1 #set the position where this tstring is in this vector
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token)) #it stores a set of tstrings sth like Identifier-@@-ID
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings))) #how many tstring in this set
        self.string_to_number = dict()
        self.number_to_string = dict() 
        max_number = 0 #counter
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number #index for each tstring
            self.number_to_string[max_number] = token_string #shortcut to find a tstring with index
            max_number += 1
        
        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                # if idx > 2 and idx < len(token_list) - 2:
                #     token_string = self.token_to_string(token)
                #     p_previous_token_string = self.token_to_string(token_list[idx - 2])
                #     previous_token_string = self.token_to_string(token_list[idx - 1])
                #     post_token_string = self.token_to_string((token_list[idx + 1]))
                #     p_post_token_string = self.token_to_string((token_list[idx + 2]))
                #     xs.append([self.one_hot(p_previous_token_string),self.one_hot(previous_token_string),self.one_hot(post_token_string),self.one_hot(p_post_token_string)])
                #     ys.append(self.one_hot(token_string))
                if idx > 1:
                    token_string = self.token_to_string(token)
                    p_previous_token_string = self.token_to_string(token_list[idx - 2])
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append([self.one_hot(p_previous_token_string), self.one_hot(previous_token_string)])
                    ys.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xs)))

        # return (array(xs), ys)
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None,2,len(self.string_to_number)])
        ## self.net = tflearn.fully_connected(self.net, 32)
        ## self.net = tflearn.fully_connected(self.net, 32)
        # self.net = tflearn.lstm(self.net, 256,return_seq=True)#
        # self.net = tflearn.dropout(self.net, 0.5);
        # self.net = tflearn.lstm(self.net, 256);
        # self.net = tflearn.dropout(self.net, 1);
        self.net = tflearn.lstm(self.net, 128, dropout=(0.8, 0.8))
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)
    
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=256, show_metric=True)
        self.model.save(model_file)
        
    def query(self, prefix, suffix):
        # if (len(prefix) > 1 and len(suffix) > 1):
        #     previous_token_string = self.token_to_string(prefix[-1])
        #     p_previous_token_string = self.token_to_string(prefix[-2])
        #     post_token_string = self.token_to_string(suffix[0])
        #     p_post_token_string = self.token_to_string((suffix[1]))
        #     x = [self.one_hot(p_previous_token_string),self.one_hot(previous_token_string),self.one_hot(post_token_string),self.one_hot(p_post_token_string)]
        #     y = self.model.predict([x]).tolist()
        #     predicted = y[0]
        #     best_number = predicted.index(max(y[0]))
        #     best_string = self.number_to_string[best_number]
        #     best_token = self.string_to_token(best_string)
        flag = True
        prob = 0#
        lastprob = 0
        b_tokens = []
        while(flag):
            if(len(prefix) > 1):
                previous_token_string = self.token_to_string(prefix[-1])
                p_previous_token_string = self.token_to_string(prefix[-2])
                x = [self.one_hot(p_previous_token_string), self.one_hot(previous_token_string)]
                y = self.model.predict([x]).tolist()
                predicted = y[0]
                p = max(y[0])#
                best_number = predicted.index(p)#
                sortedlist = sorted(predicted)
                first_number = sortedlist[0]
                second_number = sortedlist[1]
                best_string = self.number_to_string[best_number]
                best_token = self.string_to_token(best_string)
                print("p is %s"%(p))
                print("prob is %s" % (prob))
                if( (p - prob) > 0.1 and len(b_tokens) < 4 ):#
                    prefix.append(best_token)
                    b_tokens.append(best_token)
                    prob = p
                elif( p > 0.5 and prob > 0.8 and len(b_tokens)< 4):
                    prefix.append(best_token)
                    b_tokens.append(best_token)
                    prob = p
                # elif( abs(p - prob) < 0.15 and len(b_tokens) < 4 ):
                #     prefix.append(best_token)
                #     b_tokens.append(best_token)
                else:
                    flag = False
            else:
                flag = False
                best_token = self.string_to_token("Identifier-@@-ID")
                b_tokens.append(best_token)

            # else:
            #     best_token = self.string_to_token("Punctuator-@@-\"STR\"")
        # return [best_token]
        print(len(b_tokens))
        return b_tokens
