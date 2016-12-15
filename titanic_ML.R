#Rscript for Kaggle Titanic challenge 
#Based on  datacamp tutorial - https://campus.datacamp.com/courses/kaggle-r-tutorial-on-machine-learning/ and Megan Risdal - https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

#Libraries
library(dplyr)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)

# Assign the training and testing set
#train <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
#test <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))
train = read.csv("input/train.csv",stringsAsFactors = F)
train$DataId = "train"
test = read.csv("input/test.csv",stringsAsFactors = F)
test$DataId = "test"

#Create a common dataframe
df = bind_rows(train,test)

#DEALING WITH MISSING DATA
# Passenger on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
# We code all embarkment codes as factors.
df$Embarked[c(62,830)] = "S"
df$Embarked <- factor(df$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
df$Fare[1044] <- median(df$Fare, na.rm=TRUE)

#FEATURE ENGINEERING
# Create the column child, and indicate whether passenger is child or no child
df$Child <- NA
df$Child[df$Age<18] <- 1
df$Child[df$Age>=18] <- 0

#legger inn en variabel for familiestørrelse
df$family_size <- df$SibSp+df$Parch+1

#Add title-variable
df$Title <- gsub('(.*, )|(\\..*)', '', df$Name)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Reassign mlle, ms, and mme accordingly
df$Title[df$Title == 'Mlle']        <- 'Miss' 
df$Title[df$Title == 'Ms']          <- 'Miss'
df$Title[df$Title == 'Mme']         <- 'Mrs' 
df$Title[df$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex
table(df$Sex, df$Title)

#Family size
df$Surname <- sapply(df$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
df$family_id <- paste0(df$Surname,as.character(df$family_size))
nlevels(as.factor(df$family_id))
df$family_id[df$family_size <= 2] <- 'Small'
nlevels(as.factor(df$family_id))
famIDs <- data.frame(table(df$family_id))
famIDs <- famIDs[famIDs$Freq <= 2,]
df$family_id[df$family_id %in% famIDs$Var1] <- 'Small'
df$family_id = as.factor(df$family_id)

#Cabin
table(as.factor(df$Cabin),df$Pclass)
#Er bokstaven nivå i skipet eller noe?
strsplit(df$Cabin,"[1234567890]")[[1]][1]
df$cabin_letter <- sapply(df$Cabin, FUN=function(x) {strsplit(x, split="[1234567890]")[[1]][1]})
df$cabin_letter[is.na(df$cabin_letter)==T] = "0" #setter 0 for de uten cabin
mosaicplot(table(as.factor(df$cabin_letter), df$Survived), main='Cabin Letter by Survival', shade=TRUE)
df$cabin_letter= as.factor(df$cabin_letter)

#Hva med tallet? Har det betydning (og annen betydning)?
df$cabin_number <- sapply(df$Cabin, FUN=function(x) {strsplit(x, split="[ABCDEFGT]")[[1]][2]})
df$cabin_number[is.na(df$cabin_number)==T] = "0" #setter 0 for de uten cabin
nlevels(as.factor(df$cabin_number))
hist(as.integer(df$cabin_number),df$Survived)
mosaicplot(table(as.factor(df$cabin_number), df$Survived), main='Cabin Number by Survival', shade=TRUE)
mosaicplot(table(as.factor(df$cabin_number),as.factor(df$cabin_letter), df$Survived), main='Cabin Letter by Survival', shade=TRUE)

#ikke helt lett, la denne ligge inntil videre

#Kan number of cabins være noe?
length(strsplit(df$Cabin[28],"[ABCDEFGHT]")[[1]])
df$cabin_count <- sapply(df$Cabin, FUN=function(x) {length(strsplit(x, split="[1234567890]")[[1]])/2})
table(df$cabin_count)
table(df$cabin_count,df$cabin_number)
df$cabin_count[df$cabin_count==0.5&nchar(df$cabin_number)==1]=1
df$cabin_count[df$cabin_count==1.5&nchar(df$cabin_number)==3]=2
mosaicplot(table(as.factor(df$cabin_count), df$Survived), main='Number of cabins by Survival', shade=TRUE)

#recode characters into factors (?)
df$Sex = as.factor(df$Sex)
df$Title = as.factor(df$Title)

# How to fill in missing Age values?
# Done after feature engineering
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method="anova" since you are predicting a continuous variable.
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                       data=df[!is.na(df$Age),], method="anova")
df$Age[is.na(df$Age)] <- predict(predicted_age, df[is.na(df$Age),])

#show sum of na
apply(df,2,function(x){sum(is.na(x)==T)})


#MODELING!
#split the df back up
train = subset(df,DataId=="train",select=-DataId)
test = subset(df,DataId=="test",select=-c(DataId,Survived))

#First round of submissions was done without dealing with the missing values as above
# Two-way comparison for a baseline model based on sex
table(df$Sex, df$Survived)
prop.table(table(df$Sex, df$Survived),1)
prop.table(table(df$Sex, df$Survived),2)

#75 % of women in train survives, 68 % of those who survived were women
#baseline model: all women survives 
baseline = filter(df,DataId=="test")
baseline$Survived[baseline$Sex=="female"]=1
baseline$Survived[baseline$Sex=="male"]=0
write.csv(select(baseline,PassengerId,Survived),"output/my_solution_1.csv",row.names = F)

#Model 2: Decision Tree with class, sex, age, siblings/spouses, parents/children, fare, embarked
# Build the decision tree
my_tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")

# Visualize the decision tree using plot() and text()
plot(my_tree)
text(my_tree)
# Plot a fancified tree
fancyRpartPlot(my_tree,sub="Decision tree for survival on the Titanic")

#make a prediction (made a stupid error with data as an argument, instead of newdata - rtfm)
my_prediction <- predict(my_tree, newdata=test, type="class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Check that  data frame has 418 entries
nrow(my_solution)==418

# Write  a csv file 
write.csv(my_solution, file="output/my_solution_2.csv", row.names=F)

#MODEL 3: adjusting the parameters for the decision three
#overfitting: minsplit = 2 er minste leafnode mulig, cp betyr ingen splitstop. dette bør gi en overfitta modell
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                       data = train, method = "class", control = rpart.control(minsplit = 2, cp = 0))
fancyRpartPlot(my_tree_three)
my_prediction <- predict(my_tree_three, newdata=test, type="class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution, file="output/my_solution_overfitted.csv", row.names=F)
#en overfitta model vil ikke gi bedre resultat - selv om den forklarer train-settet perfekt, så kan den ikke forklare test-settet på noen særlig god måte

#MODEL 4: Decision tree with family size
my_tree_four <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, data=train, method="class")
fancyRpartPlot(my_tree_four)
my_prediction <- predict(my_tree_four, newdata=test, type="class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution, file="output/my_solution_3.csv", row.names=F)

#MODEL 5: Decision tree with  title
my_tree_five <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data=train, method="class")
fancyRpartPlot(my_tree_five)
my_prediction <- predict(my_tree_five, newdata=test, type="class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution, file="output/my_solution_4.csv", row.names=F)
#this is currently the most successful at the public leaderboard

#MODEL 6: Decision tree with family size, title and child
my_tree_six <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + family_size + Child, data=train, method="class")
fancyRpartPlot(my_tree_six)
my_prediction <- predict(my_tree_six, newdata=test, type="class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution, file="output/my_solution_5.csv", row.names=F)

#MODEL 6.1: Decision tree with title, family size, family_id, cabin_letter and cabin count
my_tree_six_one <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + family_size + family_id + cabin_letter + cabin_count, data=train, method="class")
fancyRpartPlot(my_tree_six_one)
my_prediction <- predict(my_tree_six_one, newdata=test, type="class")
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution, file="output/my_solution_6.1.csv", row.names=F)
#her dominerer title og familie-id fullstendig

#MODEL 7: Random Forest
#It grows multiple (very deep) classification trees using the training set. 
#At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. 
#For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. 
#This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.
#requires no missing data!

#One more important element in Random Forest is randomization 
#to avoid the creation of the same tree from the training set. 
#You randomize in two ways: by taking a randomized sample of the rows 
#in your training set and by only working with a limited and 
#changing number of the available variables for every node of the tree.

# Set seed for reproducibility
set.seed(111)

# Apply the Random Forest Algorithm
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age +  SibSp + Parch + Fare + Embarked + Title, data=train, importance=T, ntree=1000)

# Make your prediction using the test set
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

write.csv(my_solution, file="my_solution_6.csv", row.names=F)
varImpPlot(my_forest)

#Model 8 - logistic regression

#basert på https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
#train <- read.csv('Train_Old.csv')
#create training and validation data from given data
#install.packages('caTools')
#library(caTools)
#set.seed(88)
#split <- sample.split(train$Recommended, SplitRatio = 0.75)
#get training and test data
#dresstrain <- subset(train, split == TRUE)
#dresstest <- subset(train, split == FALSE)
#model <- glm (Survived ~ .-ID, data = dresstrain, family = binomial)
#summary(model)
#predict <- predict(model, type = 'response')

model <- glm(as.factor(Survived) ~ Pclass + Sex + Age +  SibSp + Parch + Fare + Embarked + Title + cabin_count, data = train, family = binomial)
summary(model)
predict <- predict(model, type = 'response')

apply(train,2,function(x){sum(is.na(x)==T)})

#confusion matrix
table(train$Survived, predict > 0.5)

#ROCR Curve
library(ROCR)
ROCRpred <- prediction(predict, train$Survived)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

#plot glm
library(ggplot2)
ggplot(train, aes(x=Pclass, y=Survived)) + geom_point() + 
        stat_smooth(method="glm", family="binomial", se=FALSE)

my_prediction <- predict(model, type = 'response',newdata=test)

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = round(my_prediction))
write.csv(my_solution, file="output/my_solution_7.csv", row.names=F)
