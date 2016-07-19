find ./include -name "*.hpp" | xargs git add
find ./include -name "*.h" | xargs git add
find ./src -name "*.cpp" | xargs git add
find ./src -name "*.cu" | xargs git add
find ./src -name "*Make*" | xargs git add
