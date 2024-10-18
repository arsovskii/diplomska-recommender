<script>
	import { fade, fly, slide } from 'svelte/transition';

	export let books = [
		{
			author: "['Tommy Tenney']",
			category: "['Fiction']",
			countReviews: 15,
			id: 314637,
			image:
				'http://books.google.com/books/content?id=Fj-5AgAAQBAJ&printsec=frontcover&img=1&zoom=1&edge=curl&source=gbs_api&fife=w800',
			rating: 4.4,
			title: 'Hadassah'
		}
	];
	export let onLinkClick; // Accept the function as a prop

	export let loading;
	export let inputting;
</script>

{#if inputting}
	{#if loading}
		<div class="text-center">
			<div class="loading loading-lg mx-auto"></div>
		</div>
	{:else if books.length === 0}
		<p class="text-center" transition:fade>No books found</p>
	{:else}
		{#key books}
			<div class="" transition:fade>
				<table class="table">
					<!-- head -->

					<tbody>
						<!-- row 1 -->
						{#each books as book}
							<tr class="h-20 max-h-20 hover" transition:fade>
								<a href="/book/{book.id}" on:click={onLinkClick}>
									<td>
										<div class="avatar">
											<div class="w-32 rounded">
												<img src={book.image} alt={book.title} />
											</div>
										</div>
									</td>
									<td class="text-xl stat-value text-wrap">
										{book.title}
										<br />
										<span class="badge badge-ghost badge-sm">{book.author}</span>
									</td>
								</a>
							</tr>
						{/each}
						<!-- row 2 -->
					</tbody>
				</table>
			</div>
		{/key}
	{/if}
{:else}
	<p class="text-center text-xl">Search for a book!</p>
{/if}
