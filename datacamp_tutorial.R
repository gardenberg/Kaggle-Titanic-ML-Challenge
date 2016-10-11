#Rscripts from datacamp tutorial - https://campus.datacamp.com/courses/kaggle-r-tutorial-on-machine-learning/
#also includes feature engineering from Megan Risdal - https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

#Libraries
library(dplyr)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Assign the training and testing set
#train <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
#test <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))
train = read.csv("input/train.csv",stringsAsFactors = F)
train$DataId = "train"
test = read.csv("input/test.csv",stringsAsFactors = F)
test$DataId = "test"

#Create a common dataframe
df = bind_rows(train,test)

# Create the column child, and indicate whether passenger is child or no child
df$Child <- NA
df$Child[df$Age<18] <- 1
df$Child[df$Age>=18] <- 0

# Two-way comparison
table(df$Child, df$Survived)
prop.table(table(df$Child, df$Survived),1)
table(df$Sex, df$Survived)
prop.table(table(df$Sex, df$Survived),1)
prop.table(table(df$Sex, df$Survived),2)

#75 % of women in train survives, 68 % of those who survived were women
#baseline model: all women survives 
baseline = filter(df,DataId=="test")
baseline$Survived[baseline$Sex=="female"]=1
baseline$Survived[baseline$Sex=="male"]=0
write.csv(select(baseline,PassengerId,Survived),"output/my_solution_1.csv",row.names = F)

# Decision Trees
#split the df back up
train = subset(df,DataId=="train",select=-DataId)
test = subset(df,DataId=="test",select=-c(DataId,Survived))

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
write.csv(my_solution, file="my_datacamp2_solution.csv", row.names=F)

# Create a new decision tree my_tree_three
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class", control=rpart.control(minsplit=50, cp=0) )

#overfitting
#minsplit = 2 er minste leafnode mulig, cp betyr ingen splitstop. dette bør gi en overfitta modell
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                       data = train, method = "class", control = rpart.control(minsplit = 2, cp = 0))

# Visualize my_tree_three
fancyRpartPlot(my_tree_three)

#minsplit = 2 er minste leafnode mulig, cp betyr ingen splitstop. dette bør gi en overfitta modell
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                       data = train, method = "class", control = rpart.control(minsplit = 50, cp = 0))

# Visualize my_tree_three
fancyRpartPlot(my_tree_three)

#feature engineering - legger inn en variabel for familiestørrelse
train_two <- train

train_two$family_size <- train_two$SibSp+train_two$Parch+1

# Create a new decision tree 
my_tree_four <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, data=train_two, method="class")

#Add title-variable
train_new=train
test_new=test
train_new$Title <- gsub('(.*, )|(\\..*)', '', train_new$Name)
test_new$Title <- gsub('(.*, )|(\\..*)', '', test_new$Name)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
train_new$Title[train_new$Title == 'Mlle']        <- 'Miss' 
train_new$Title[train_new$Title == 'Ms']          <- 'Miss'
train_new$Title[train_new$Title == 'Mme']         <- 'Mrs' 
train_new$Title[train_new$Title %in% rare_title]  <- 'Rare Title'
test_new$Title[test_new$Title == 'Mlle']        <- 'Miss' 
test_new$Title[test_new$Title == 'Ms']          <- 'Miss'
test_new$Title[test_new$Title == 'Mme']         <- 'Mrs' 
test_new$Title[test_new$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(train_new$Sex, train_new$Title)

str(train_new)
str(test_new$Title)

# Create a new decision tree
my_tree_five <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data=train_new, method="class")

# Make your prediction using `my_tree_five` and `test_new`
my_prediction <- predict(my_tree_five, test_new, type="class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test_new$PassengerId, Survived = my_prediction)

# Check that your data frame has 418 entries
nrow(my_solution)

# Write your solution to a csv file with the name my_solution.csv
write.csv(my_solution, file="my_datacamp3_solution.csv", row.names=F)


# Random Forest
#It grows multiple (very deep) classification trees using the training set. 
#At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. 
#For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. 
#This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.
#requires no missing data!

# cleaning up
# All data, both training and test set
all_data = rbind(train_new,test_new)

# Passenger on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
# We code all embarkment codes as factors.
all_data$Embarked[c(62,830)] = "S"
all_data$Embarked <- factor(combi$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
all_data$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method="anova" since you are predicting a continuous variable.
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                       data=all_data[!is.na(all_data$Age),], method="anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])

# Split the data back into a train set and a test set
train <- all_data[1:891,]
test <- all_data[892:1309,]

#One more important element in Random Forest is randomization to avoid the creation of the same tree from the training set. You randomize in two ways: by taking a randomized sample of the rows in your training set and by only working with a limited and changing number of the available variables for every node of the tree.

# Load in the package
library(randomForest)

# Train set and test set
str(train)
str(test)

# Set seed for reproducibility
set.seed(111)

# Apply the Random Forest Algorithm
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age +  SibSp + Parch + Fare + Embarked + Title, data=train, importance=T, ntree=1000)

# Make your prediction using the test set
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

write.csv(my_solution, file="my_datacamp4_solution.csv", row.names=F)

#not better than datacamp3, actually...

varImpPlot(my_forest)
