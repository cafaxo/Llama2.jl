using Llama2
using Test
using IOCapture

if !isdir(joinpath("..", "bin"))
  mkdir(joinpath("..", "bin"))
end

storiesbin = joinpath("..", "bin", "stories42M.bin")
tokenizerbin = joinpath("..", "bin", "tokenizer.bin")

downloader = Base.Downloads()
function download_if_missing(url, path)
  if !isfile(path)
    downloader.download(url, path)
  end
end

@sync begin
  @async download_if_missing("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", storiesbin)
  @async download_if_missing("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", tokenizerbin)
end

model = load_karpathy_model(storiesbin, tokenizerbin)

sample_1 = (
  prompt="Once upon a time, in a land far, far away, there was a ",
  output="""Once upon a time, in a land far, far away, there was a icy hill. On top of the hill, there was a big tree. Under the tree, there was a small house. In the house, there lived a nice lady named Sue. Sue liked to cook yummy food for her friends.
One day, Sue wanted to cook a big meal for her friends. She went to the icy hill to find some snow. She put the snow on the ground and made a big snowball. Then, she put the snowball in the sun to make it warm.
Sue's friends came to her house. They saw the big snowball and were very happy. They all sat down and ate the yummy food Sue cooked. They laughed and talked while they ate. After they finished eating, they played on the icy hill until the sun went down.
"""
)

sample_2 = (
  prompt="What are the main differences between TCP and UDP protocols?",
  output="""What are the main differences between TCP and UDP protocols? They were both very different. One day, Upsy and Upsy were playing together. Upsy was very happy and they were having lots of fun.
Suddenly, Upsy stopped and looked at Upsy. Upsy said, "Let's play a game. I'll hide and you try to find me." Upsy agreed and started to look for Upsy. Upsy found Upsby and Upsy said, "You found me!"
 Upsy was so happy that Upsy had found her. Upsy said, "Let's play again!" Upsy said, "Yes, let's play again!" They played the game for a long time and had lots of fun.
At the end of the day, Upsy said, "That was so much fun! Let's play again tomorrow." Upsy said, "Yes, let's play again tomorrow!" They both smiled and went home."""
)

sample_3 = (
  prompt="Write a poem about the changing seasons",
  output="""Write a poem about the changing seasons. She was so excited to learn about the seasons. She wanted to learn more about them.
One day, she asked her mom, "What season is it?"
Her mom smiled and said, "It's fall. The leaves change color and fall off the trees."
D connected to the trees and said, "That's right! The seasons change because it's autumn."
D connected to the trees and said, "That's so cool!"
Dad was watching and said, "That's so cool! I'm so proud of you for learning about the seasons."
Dad was so proud of his daughter. He said, "You're so smart, my little one!"
Dad was so proud of his daughter. He hugged her and said, "I'm so proud of you!"
Dad was so proud of his daughter. He was so proud of her. He was proud of her for learning about the seasons."""
)

sample_4 = (
  prompt="How do you bake a chocolate cake from scratch?",
  output="""How do you bake a chocolate cake from scratch?
Mum said, "Let's go to the store and get some eggs and flour."
So they went to the store and bought some eggs and flour.
When they got home, Mum cracked the eggs and put them in a bowl. She added some flour and stirred it all together.
Mum said, "Now we need to put the mix in the oven."
Mum put the mix in the oven and set the timer.
Mum said, "Now we have to wait for the cake to bake."
Mum and Dad waited and waited. Finally, the timer beeped and the cake was ready.
Mum said, "Let's take the cake out of the oven."
Mum and Dad took the cake out and put it on the table.
Mum said, "Let's cut the cake into pieces."
Mum and Dad cut the cake into pieces and put them on plates.
Mum said, "Let's eat the cake!"
Mum and Dad ate the cake and it was delicious.
Mum said, "Let's bake another cake tomorrow!"
Mum and Dad smiled and said, "Yes, let's bake another cake tomorrow!"
"""
)

sample_5 = (
  prompt="What are the top 5 tourist attractions in Paris?",
  output="""What are the top 5 tourist attractions in Paris? It's a very special day for us.
Mum and Dad were so excited. They had been planning this special day for a long time.
Mum said, "Let's go to the top and take a look!"
Dad said, "Yes, let's go!"
When they arrived at the top, they were amazed. It was so high up!
Mum said, "Let's take a break and have a snack."
Dad said, "That's a great idea!"
They all sat down and ate some delicious snacks.
Mum said, "This is the best day ever!"
Dad said, "Yes, it's a very special day for us."
Mum and Dad had a great time at the top. They were so happy to be together."""
)

sample_6 = (
  prompt="Explain the theory of relativity in simple terms",
  output="""Explain the theory of relativity in simple terms. She was always so busy, but she never had time to play. One day, she was walking in the park and noticed a big, red balloon. She was so excited and ran over to it. She wanted to play with it, but it was too high up in the sky.
Suddenly, a voice called out from behind her. It was her mom. "Come here, sweetheart," she said. "Let's go home."
Even though she was tired, she followed her mom. When they got home, her mom said, "Let's go to the park and play with the balloon."
Eenen was so excited. She ran to the park and saw the balloon. She ran up to it and grabbed it. She was so happy.
But then, the wind started to blow. The balloon flew away and she couldn't catch it. She started to cry.
Eenen's mom hugged her and said, "It's okay, sweetheart. We can get another balloon."
Eenen smiled and said, "Okay, mommy."
But the next day, when they went to the park, the balloon was gone. It had flown away and was gone forever.
Eenen was so sad. She had wanted to play with the balloon, but it was gone. She had been so busy playing that she had forgotten to take care of the balloon."""
)

samples = [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6]

@testset "stories42m sampling at temperature 0.0" begin
  for s in samples
    @test begin
      ___, output = IOCapture.capture() do
        sample(model, s.prompt, temperature=zero(Float32))
      end
      occursin(s.output, output)
    end
  end
end