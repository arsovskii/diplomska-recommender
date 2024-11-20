<script lang="ts">
	import { onMount } from 'svelte';

	let genres = [
		'Fantasy',
		'Science Fiction',
		'Mystery',
		'Romance',
		'Thriller',
		'Historical Fiction',
		'Young Adult (YA)',
		'Non-Fiction',
		'Graphic Novels & Comics',
		'Self-Help',
		'Other'
	];

	let selectedGenres = new Set();
	let isSaveDisabled = true;

	let init_modal: HTMLDialogElement;
	onMount(() => {
		init_modal.showModal();
	});

	function saveGenre() {
		if (selectedGenres.size > 0) {
			// Зачувај ги избраните жанрови
			localStorage.setItem('favoriteGenres', JSON.stringify(Array.from(selectedGenres)));
			init_modal.close();
		}
	}

	function toggleGenre(genre: string) {
		console.log(genre);
		if (selectedGenres.has(genre)) {
			selectedGenres.delete(genre);
		} else {
			selectedGenres.add(genre);
		}
		console.log(selectedGenres);
		isSaveDisabled = selectedGenres.size === 0;
	}
</script>

<dialog id="init_modal" class="modal" bind:this={init_modal}>
	<div class="modal-box w-11/12 max-w-md text-lg">
		<h3 class=" text-3xl font-bold">Hi!</h3>
		<p class="py-4">Please select your favourite genres!</p>

		<div>
			{#each genres as genre}
				<div class="form-control">
					<label class="label cursor-pointer justify-start">
						<input
							type="checkbox"
							class="checkbox checkbox-secondary"
							checked={selectedGenres.has(genre)}
							on:change={() => toggleGenre(genre)}
						/>
						<span class="label-text mx-3 text-lg">{genre}</span>
					</label>
				</div>
			{/each}
		</div>
		<div class="modal-action">
			<form method="dialog">
				<!-- if there is a button in form, it will close the modal -->
				<button
					class="btn btn-secondary text-secondary-content"
					on:click={saveGenre}
					disabled={isSaveDisabled}>Save</button
				>
			</form>
		</div>
	</div>
</dialog>
