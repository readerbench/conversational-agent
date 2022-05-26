# import os
# from actions.utils import get_grakn_entities
#
#
# def write_to_file(file_name, entities):
#     os.makedirs(os.path.dirname(file_name), exist_ok=True)
#     with open(file_name, "w+", encoding="utf-8") as f:
#         for e in entities:
#             f.write(f"{e}\n")
#
#
# def main():
#     courses = get_grakn_entities("course")
#     write_to_file("microworlds/university_guide/data/lookup/course.txt", set(map(lambda x: x["name"], courses)))
#
#     professors = get_grakn_entities("professor")
#     write_to_file("microworlds/university_guide/data/lookup/professor.txt", set(map(lambda x: x["name"], professors)))
#
#     rooms = get_grakn_entities("room")
#     write_to_file("microworlds/university_guide/data/lookup/room.txt", set(map(lambda x: x["name"], rooms)))
#
#
# if __name__ == "__main__":
#     main()
