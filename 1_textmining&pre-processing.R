##################################################
########### First Feature - Total Score ##########
##################################################

library("plyr") 
library("stringr") 
library("SnowballC")

df.all <- read.csv("news_headline.csv")
df.all$year <- substring(df.all$publish_date,1,4)
df.all$month <- substring(df.all$publish_date,5,6)
df.all$day <- substring(df.all$publish_date,7,8)
df <- df.all[df.all$year %in% c("2011","2012","2013","2014","2015"),]

df_test <- df.all[df.all$year %in% c("2016","2017"),]

headline_test <- as.character(df_test$headline_text)
headline <- as.character(df$headline_text)


########## Pos/Neg sentiment score ##########
#Load dictionary
library(readxl)
dict <- read_excel("posnegDictionary.xls")
dict.pos <- na.omit(dict[c(1,3)])
dict.neg <- na.omit(dict[c(1,4)])

pos <- as.character(dict.pos$Entry)
neg <- as.character(dict.neg$Entry)

# remove punctuation (predefined classes of characters)
pos = gsub("[[:punct:]]", "", pos)
neg = gsub("[[:punct:]]", "", neg)
# remove digits?
pos = gsub('\\d+', '', pos)
neg = gsub('\\d+', '', neg)
# tolower
pos = tolower(pos)
neg = tolower(neg)


score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
        # Parameters
        # sentences: vector of text to score
        # pos.words: vector of words of postive sentiment
        # neg.words: vector of words of negative sentiment
        # .progress: passed to laply() to control of progress bar. “text” for old 
        # school ASCII in the console window. Default is “none”.
        
        # create simple array of scores with laply
        # do not confuse laply with the lapply function
        scores = laply(sentences,
                       function(sentence, pos.words, neg.words)
                       {
                               # remove punctuation (predefined classes of characters)
                               sentence = gsub("[[:punct:]]", "", sentence)
                               # remove control characters
                               sentence = gsub("[[:cntrl:]]", "", sentence)
                               # remove digits?
                               sentence = gsub('\\d+', '', sentence)
                               
                               # define error handling function when trying tolower
                               tryTolower = function(x)
                               {
                                       # create missing value
                                       y = NA
                                       # tryCatch error
                                       try_error = tryCatch(tolower(x), error=function(e) e)
                                       # if not an error
                                       if (!inherits(try_error, "error"))
                                               y = tolower(x)
                                       # result
                                       return(y)
                               }
                               # use tryTolower with sapply 
                               sentence = sapply(sentence, tryTolower)
                               
                               # split sentence into words with str_split (stringr package)
                               word.list = str_split(sentence, "\\s+")
                               words = unlist(word.list)
                               
                               # compare words to the dictionaries of positive & negative terms
                               # match("s",c("a","s"))
                               # match("b",c("a","s"))
                               pos.matches = match(words, pos.words)
                               neg.matches = match(words, neg.words)
                               
                               # get the position of the matched term or NA
                               # we just want a TRUE/FALSE
                               pos.matches = !is.na(pos.matches)
                               neg.matches = !is.na(neg.matches)
                               
                               score = sum(pos.matches) - sum(neg.matches)
                               return(score)
                       }, pos.words, neg.words, .progress=.progress )
        
        # data frame with scores for each sentence
        scores.df = data.frame(text=sentences, score=scores)
        return(scores.df)
}

# apply function score.sentiment
scores <- score.sentiment(headline, pos, neg, .progress='text')
scores_test <- score.sentiment(headline_test,pos,neg, .progress ='text')

df$score <- scores$score
df_test$score <- scores_test$score



########## company relevancy score ##########
company.sentiment = function(df, pos.words, .progress='none')
{
        sentences <- df$headline_text
        scores = laply(sentences,
                       function(sentence, pos.words)
                       {
                               # remove punctuation (predefined classes of characters)
                               sentence = gsub("[[:punct:]]", "", sentence)
                               # remove control characters
                               sentence = gsub("[[:cntrl:]]", "", sentence)
                               # remove digits?
                               sentence = gsub('\\d+', '', sentence)
                               
                               # define error handling function when trying tolower
                               tryTolower = function(x)
                               {
                                       # create missing value
                                       y = NA
                                       # tryCatch error
                                       try_error = tryCatch(tolower(x), error=function(e) e)
                                       # if not an error
                                       if (!inherits(try_error, "error"))
                                               y = tolower(x)
                                       # result
                                       return(y)
                               }
                               # use tryTolower with sapply 
                               sentence = sapply(sentence, tryTolower)
                               
                               # split sentence into words with str_split (stringr package)
                               word.list = str_split(sentence, "\\s+")
                               words = unlist(word.list)
                               
                               # if find any billiton dictonary , set to 1
                               pos.matches = match(words, pos.words)
                               pos.matches = !is.na(pos.matches)
                               isCompany = ifelse(sum(pos.matches)>0,1,0)
                               
                               return(isCompany)
                       }, pos.words, .progress=.progress )
        
        # data frame with scores for each sentence
        company.df = data.frame(df, company=scores)
        return(company.df)
}

bph <- readLines("billitonDictionary.txt")
bph <- gsub('[\r\n\t]', '', bph)

df_companyscore = company.sentiment(df, bph, .progress='text')
df_companyscore_test <- company.sentiment(df_test, bph, .progress='text')


########## total score ##########
df_companyscore$total_score <- df_companyscore$score * df_companyscore$company
df_companyscore_test$total_score <- df_companyscore_test$score * df_companyscore_test$company


# combine with y
df.y <- read.csv("stock_5newFeatures.csv")
colnames(df.y)[3] <- "publish_date"

df.y.train <- df.y[df.y$year %in% c("2011","2012","2013","2014","2015"),c(3,12:18)]
df.y.test <- df.y[df.y$year %in% c("2016","2017"),c(3,12:18)]


##################################################
########### aggregate data for each day ##########
##################################################

# aggregate total score
df_feature1 <- aggregate(df_companyscore[,6:8],list(df_companyscore$publish_date),mean)
colnames(df_feature1)[1] <- "publish_date"
df_feature1[2:3] <- NULL

df_feature1_test <- aggregate(df_companyscore_test[,6:8],list(df_companyscore_test$publish_date),mean)
colnames(df_feature1_test)[1] <- "publish_date"
df_feature1_test[2:3] <- NULL

# aggregate headlines daily - topic, terms
df_topic <- ddply(df, .(publish_date), summarise, headline_text = paste0(headline_text,collapse=","))
df_topic_test <- ddply(df_test, .(publish_date), summarise, headline_text = paste0(headline_text,collapse=","))


##################################################
########### Second Feature - topic      ##########
##################################################

########## topic modelling ##########
# topic analysis
library(topicmodels)
content <- as.character(df_topic$headline_text)
content_test <- as.character(df_topic_test$headline_text)
custom_stopwords = ""
contentToDtm <- function(content){
        docs <- VCorpus(VectorSource(content))
        
        #Transform to lower case
        docs <-tm_map(docs,content_transformer(tolower))
        #remove punctuation
        docs <- tm_map(docs, removePunctuation)
        #Strip digits
        docs <- tm_map(docs, removeNumbers)
        #remove stopwords
        docs <- tm_map(docs, removeWords, stopwords("english"))
        #remove whitespace
        docs <- tm_map(docs, stripWhitespace)
        #Stem document
        docs <- tm_map(docs, stemDocument)
        
        docs <- tm_map(docs,removeWords,c(stopwords("english"),custom_stopwords))
        docs <- tm_map(docs, PlainTextDocument)
        #dtm_temp <- DocumentTermMatrix(docs)
        return(docs)
}

# checking the raw words freq
library(wordcloud)
docs <- contentToDtm(content)
tdm <- TermDocumentMatrix(docs)
m <- as.matrix(tdm)
v <- sort(rowSums(m), decreasing = TRUE)
d <- data.frame(word = names(v), freq = v)
head(d, 10)
set.seed(100)
png("wordcloud_b4.png")
wordcloud(words = d$word, freq = d$freq,  min.freq =1,
          max.words = 200, random.order = FALSE, rot.per = 0.35,
          colors = brewer.pal(8,"Dark2"))

dev.off()

# generating stopwords
stopwords = ""
for(i in nrow(d)){
        if(d[i,2]>1500){
                stopwords = c(stopwords,as.character(d[i,1]))
        }
}
custom_stopwords <- stopwords

# running the topic analysis model
docs <- contentToDtm(content)
dtm <- DocumentTermMatrix(docs)
docs_test <- contentToDtm(content_test)
dtm_test <- DocumentTermMatrix(docs_test)

# save frequently-appearing terms to a character vector
freq_dtm <- findFreqTerms(dtm,80,140)

# Create DTMs with only the frequent terms
freq_dtm_train <- dtm[, freq_dtm]
freq_dtm_test <- dtm_test[,dtm_test$dimnames$Terms %in% freq_dtm]

rowTotals <- apply(freq_dtm_train,1,sum)
freq_dtm_train_new <- freq_dtm_train[rowTotals>0,]

# tune for best topic number k
topics <- c(2:10)
D <- nrow(freq_dtm_train_new) 
FOLDS <- 5
folding <-sample(rep(1:FOLDS, D),D)
perplexity_record=matrix(data = NA, nrow = FOLDS, ncol = length(topics))
for (i in 1:length(topics)) { 
        for (fold in 1:FOLDS) { 
                ldaOut <-LDA(freq_dtm_train_new[folding != fold,],control = list(seed=100),topics[i])
                perplexity_record[fold,i] <- perplexity(ldaOut, freq_dtm_train_new[folding == fold,])
        }
}
k.index <- which.min(colMeans(perplexity_record))

k <- topics[k.index]
lda <- LDA(freq_dtm_train_new, control = list(seed=100), k)
as.matrix(terms(lda,40))

########## drawing wordcloud for LDA ##########
library(tidyverse)
library(tidytext)
library(reshape2)
lda_topics <- tidy(lda, matrix = "beta")
lda_top_terms <- lda_topics %>%
        group_by(topic) %>%
        top_n(200,beta) %>%
        ungroup() %>%
        arrange(topic,-beta)
png("wordcloud_LDA.png",width = 1280, height = 800)
lda_top_terms %>%
        mutate(topic = paste("topic",topic)) %>%
        acast(term ~ topic, value.var = "beta", fill = 0) %>%
        comparison.cloud(colors = c("#F8766D", "#00BFC4"),
                         max.words=100)
dev.off()

# topic freqency for each document
gammaDF_test <- as.data.frame(posterior(lda,freq_dtm_test)$topics)
rownames(gammaDF_test) <- 1:nrow(gammaDF_test)
gammaDF <- as.data.frame(lda@gamma) 

names(gammaDF) <- c(1:k)
names(gammaDF_test) <- c(1:k)

toptopics <- as.data.frame(cbind(document = row.names(gammaDF),topic = apply(gammaDF,1,function(x) names(gammaDF)[which(x==max(x))])))
toptopics_test <- as.data.frame(cbind(document = row.names(gammaDF_test),topic = apply(gammaDF_test,1,function(x) names(gammaDF_test)[which(x==max(x))])))

df_topic1 <- df_topic[-1470,]
df_topic1$topic <- toptopics$topic

df_topic_test1 <-  df_topic_test
df_topic_test1$topic <- toptopics_test$topic


##################################################
###### Third Feature - term freqency matrx   #####
##################################################

# convert counts to a factor
convert_counts <- function(x) {
        x <- ifelse(x > 0, x, x)
}

# apply() convert_counts() to columns of train/test data
train <- apply(freq_dtm_train_new, MARGIN = 2, convert_counts)
test <- apply(freq_dtm_test, MARGIN = 2, convert_counts)
test.rest <- as.data.frame(matrix(0, nrow(test), length(freq_dtm[!(freq_dtm %in% dtm_test$dimnames$Terms)]),
                                  dimnames=list(c(), freq_dtm[!(freq_dtm %in% dtm_test$dimnames$Terms)])),
                           stringsAsFactors=F)
test <- cbind(test, test.rest)



##################################################
####### final whole dataset combination    #######
##################################################

df_matrix_pre <- cbind(df_topic1, train)
df_matrix_test_pre <- cbind(df_topic_test1, test)

df.feature <- merge(df_matrix_pre, df_feature1, by = "publish_date", all.x = TRUE)
df.feature_test <- merge(df_matrix_test_pre, df_feature1_test, by = "publish_date", all.x = TRUE)

# write.csv(df.feature,"df.feature.csv")
# write.csv(df.feature_test,"df.feature_test.csv")

# shift column with excel
df_matrix <- read.csv("df.feature.csv")
df_matrix_test <- read.csv("df.feature_test.csv")

df_final <- merge(df.y.train, df_matrix, by = "publish_date", all.x = TRUE)
df_final_test <- merge(df.y.test, df_matrix_test, by = "publish_date", all.x = TRUE)

df_final <- na.omit(df_final)
df_final_test <- na.omit(df_final_test)

per=1
df_final$ynew=as.factor(ifelse(df_final$y<(-per),"0",ifelse(df_final$y>per,"1","2")))
df_final_test$ynew=as.factor(ifelse(df_final_test$y<(-per),"0",ifelse(df_final_test$y>per,"1","2")))

# write.csv(df_final,"data_train.csv")
# write.csv(df_final_test,"data_test.csv")


##################################################
########### Variable Selection          ##########
##################################################

datatrain=read.csv("data_train.csv")
datatest=read.csv("data_test.csv")

trainy=as.factor(datatrain[,c("ynew")])
testy=as.factor(datatest[,c("ynew")])
trainx=datatrain[,-c(which(colnames(datatrain) %in% c("ynew")))]
testx=datatest[,-c(which(colnames(datatest) %in% c("ynew")))]
trainx=trainx[,-seq(1:9)]
testx=testx[,-seq(1:9)]
traindf=cbind(trainx,trainy)

library(caret)
library(randomForest)
set.seed(333)
mod_rf=randomForest(trainy~.,traindf,importance=TRUE)
varImpPlot(mod_rf,type=2,n.var=37)
varimpt=mod_rf$importance[which(mod_rf$importance[,c("MeanDecreaseGini")]>1),]
nrow(varimpt)
left=rownames(varimpt)
trainx_red=trainx[,which(colnames(trainx) %in% left)]
testx_red=testx[,which(colnames(testx) %in% left)]
traindf_red=cbind(trainx_red,trainy)
testdf_red=cbind(testx_red,testy)
# write.csv(traindf_red,"train_final.csv")
# write.csv(testdf_red,"test_final.csv")


