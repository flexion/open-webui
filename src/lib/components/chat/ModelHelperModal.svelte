<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { config, models } from '$lib/stores';
	import { generateModelRecommendation } from '$lib/apis';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	import Modal from '../common/Modal.svelte';
	import Sparkles from '../icons/Sparkles.svelte';
	import Spinner from '../common/Spinner.svelte';
	import CodeBracket from '../icons/CodeBracket.svelte';
	import Photo from '../icons/Photo.svelte';
	import PenAlt from '../icons/PenAlt.svelte';
	import ChatBubbleOval from '../icons/ChatBubbleOval.svelte';
	import ChartBar from '../icons/ChartBar.svelte';
	import GlobeAlt from '../icons/GlobeAlt.svelte';
	import LightBulb from '../icons/LightBulb.svelte';
	import ChevronDown from '../icons/ChevronDown.svelte';
	import ChevronRight from '../icons/ChevronRight.svelte';

	const i18n = getContext('i18n');

	export let show = false;
	export let selectedModels: string[] = [];

	let activeTab: 'ask' | 'browse' = 'ask';
	let taskDescription = '';
	let loading = false;
	let recommendations: { model_id: string; reason: string }[] = [];
	let hasSearched = false;
	let expandedCategories: Record<string, boolean> = {};

	type Category = {
		key: string;
		label: string;
		icon: any;
		patterns: RegExp[];
		capKey?: string;
		description: string;
	};

	const CATEGORIES: Category[] = [
		{
			key: 'coding',
			label: 'Coding & Development',
			icon: CodeBracket,
			patterns: [/code|coder|codex|starcoder|deepseek|codestral|qwen.*coder/i],
			description: 'Models optimized for writing, reviewing, and debugging code'
		},
		{
			key: 'creative',
			label: 'Creative Writing',
			icon: PenAlt,
			patterns: [/creative|write|writer|story|novel|poet/i],
			description: 'Models suited for storytelling, copywriting, and creative content'
		},
		{
			key: 'image',
			label: 'Image Generation',
			icon: Photo,
			patterns: [/imagen|dall-?e|stable.?diff|flux|midjourney|image.*gen|🎨/i],
			capKey: 'image_generation',
			description: 'Models that can create and edit images from text prompts'
		},
		{
			key: 'video',
			label: 'Video Generation',
			icon: Photo,
			patterns: [/veo|video|🎬/i],
			capKey: 'video_generation',
			description: 'Models that can generate video clips from text or images'
		},
		{
			key: 'vision',
			label: 'Vision & Multimodal',
			icon: Photo,
			patterns: [/vision|visual|multimodal|gemini|gpt-?4o|claude.*3|pixtral/i],
			capKey: 'vision',
			description: 'Models that can understand and analyze images alongside text'
		},
		{
			key: 'analysis',
			label: 'Data & Analysis',
			icon: ChartBar,
			patterns: [/analy|data|math|reason|think/i],
			description: 'Models strong at reasoning, math, data analysis, and structured tasks'
		},
		{
			key: 'conversation',
			label: 'General Conversation',
			icon: ChatBubbleOval,
			patterns: [/chat|instruct|assist|general/i],
			description: 'Versatile models for everyday questions, brainstorming, and chat'
		},
		{
			key: 'research',
			label: 'Research & Knowledge',
			icon: LightBulb,
			patterns: [/search|research|scholar|knowledge|rag|retrieval/i],
			description: 'Models suited for research, summarization, and knowledge tasks'
		},
		{
			key: 'translation',
			label: 'Translation & Language',
			icon: GlobeAlt,
			patterns: [/translat|language|multilingual|nllb|mbart/i],
			description: 'Models specialized for translation and multilingual tasks'
		}
	];

	function categorizeModels(
		modelList: any[]
	): { category: Category; models: any[] }[] {
		const categorized: Record<string, any[]> = {};
		const assigned = new Set<string>();

		for (const cat of CATEGORIES) {
			categorized[cat.key] = [];
		}

		for (const m of modelList) {
			const haystack = [
				m.id,
				m.name ?? '',
				m?.info?.meta?.description ?? '',
				m?.info?.base_model_id ?? '',
				m?.info?.params?.system ?? ''
			]
				.join(' ')
				.toLowerCase();

			const caps = m?.info?.meta?.capabilities ?? {};

			for (const cat of CATEGORIES) {
				if (cat.capKey && caps[cat.capKey]) {
					categorized[cat.key].push(m);
					assigned.add(m.id);
					continue;
				}
				for (const pat of cat.patterns) {
					if (pat.test(haystack)) {
						if (!assigned.has(m.id) || cat.key !== 'conversation') {
							categorized[cat.key].push(m);
							assigned.add(m.id);
						}
						break;
					}
				}
			}

			// Put unmatched models in conversation as a fallback
			if (!assigned.has(m.id)) {
				categorized['conversation'].push(m);
				assigned.add(m.id);
			}
		}

		return CATEGORIES.map((cat) => ({ category: cat, models: categorized[cat.key] })).filter(
			(entry) => entry.models.length > 0
		);
	}

	function toggleCategory(key: string) {
		expandedCategories[key] = !expandedCategories[key];
	}

	$: categorizedModels = categorizeModels($models);

	async function handleSubmit() {
		if (!taskDescription.trim()) return;
		if ($models.length === 0) {
			toast.error($i18n.t('No models available'));
			return;
		}

		loading = true;
		recommendations = [];
		hasSearched = true;

		const defaultModelId = $config?.default_models?.split(',')[0]?.trim();
		const modelId = defaultModelId || selectedModels[0] || $models[0]?.id;
		if (!modelId) {
			toast.error($i18n.t('No model available for recommendation'));
			loading = false;
			return;
		}

		const availableModels = $models.map((m) => {
			const meta = m?.info?.meta ?? {};
			const entry: Record<string, string> = {
				id: m.id,
				name: m.name ?? m.id
			};
			if (meta.description) entry.description = meta.description;
			if (m?.owned_by) entry.owned_by = m.owned_by;
			if (m?.preset) entry.type = 'custom_model_or_pipe';
			if (m?.info?.base_model_id) entry.base_model_id = m.info.base_model_id;
			if (meta.capabilities) entry.capabilities = JSON.stringify(meta.capabilities);
			if (m?.info?.params?.system)
				entry.system_prompt_hint = m.info.params.system.substring(0, 200);
			return entry;
		});

		try {
			const result = await generateModelRecommendation(
				localStorage.token,
				modelId,
				taskDescription,
				availableModels
			);

			const validModelIds = new Set($models.map((m) => m.id));
			recommendations = result.filter((r) => validModelIds.has(r.model_id));
		} catch (e) {
			console.error(e);
			toast.error($i18n.t('Failed to get model recommendations'));
		} finally {
			loading = false;
		}
	}

	function selectModel(modelId: string) {
		selectedModels = [modelId];
		show = false;
		taskDescription = '';
		recommendations = [];
		hasSearched = false;
		activeTab = 'ask';
		expandedCategories = {};
	}

	function reset() {
		taskDescription = '';
		recommendations = [];
		hasSearched = false;
	}
</script>

<Modal bind:show size="md">
	<div class="px-6 py-5">
		<div class="flex items-center gap-2 mb-4">
			<Sparkles className="size-5 text-yellow-500" />
			<h3 class="text-lg font-semibold">{$i18n.t('Model Helper')}</h3>
		</div>

		<!-- Tab switcher -->
		<div class="flex gap-1 mb-4 p-0.5 rounded-xl bg-gray-100 dark:bg-gray-800">
			<button
				class="flex-1 px-3 py-1.5 text-sm font-medium rounded-lg transition {activeTab === 'ask'
					? 'bg-white dark:bg-gray-700 shadow-sm'
					: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'}"
				on:click={() => (activeTab = 'ask')}
			>
				{$i18n.t('Ask for Recommendation')}
			</button>
			<button
				class="flex-1 px-3 py-1.5 text-sm font-medium rounded-lg transition {activeTab === 'browse'
					? 'bg-white dark:bg-gray-700 shadow-sm'
					: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'}"
				on:click={() => (activeTab = 'browse')}
			>
				{$i18n.t('Browse by Use Case')}
			</button>
		</div>

		{#if activeTab === 'ask'}
			<p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
				{$i18n.t('Describe what you want to do and get a model recommendation.')}
			</p>

			<form on:submit|preventDefault={handleSubmit}>
				<div class="flex gap-2">
					<input
						type="text"
						bind:value={taskDescription}
						placeholder={$i18n.t('e.g., Help me write Python code...')}
						class="flex-1 px-3 py-2 text-sm rounded-xl border border-gray-200 dark:border-gray-700 bg-transparent outline-none focus:ring-1 focus:ring-gray-300 dark:focus:ring-gray-600"
						disabled={loading}
					/>
					<button
						type="submit"
						class="px-4 py-2 text-sm font-medium rounded-xl bg-black text-white dark:bg-white dark:text-black hover:opacity-80 transition disabled:opacity-50"
						disabled={loading || !taskDescription.trim()}
					>
						{#if loading}
							<Spinner className="size-4" />
						{:else}
							{$i18n.t('Ask')}
						{/if}
					</button>
				</div>
			</form>

			{#if loading}
				<div class="flex items-center justify-center py-8">
					<Spinner className="size-6" />
				</div>
			{:else if recommendations.length > 0}
				<div class="mt-4 space-y-2">
					{#each recommendations as rec}
						{@const model = $models.find((m) => m.id === rec.model_id)}
						<button
							class="w-full text-left p-3 rounded-xl border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-850 transition cursor-pointer flex items-center gap-3"
							on:click={() => selectModel(rec.model_id)}
						>
							<img
								src={`${WEBUI_API_BASE_URL}/models/model/profile/image?id=${encodeURIComponent(rec.model_id)}&lang=${$i18n.language}`}
								alt={model?.name ?? rec.model_id}
								class="rounded-full size-8 flex-shrink-0 object-cover"
								loading="lazy"
							/>
							<div class="min-w-0">
								<div class="font-medium text-sm">{model?.name ?? rec.model_id}</div>
								<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{rec.reason}</div>
							</div>
						</button>
					{/each}
				</div>

				<div class="mt-3 flex justify-center">
					<button
						class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
						on:click={reset}
					>
						{$i18n.t('Ask Again')}
					</button>
				</div>
			{:else if hasSearched}
				<div class="mt-4 text-center text-sm text-gray-500 dark:text-gray-400 py-4">
					{$i18n.t('No recommendations found. Try a different description.')}
				</div>
				<div class="flex justify-center">
					<button
						class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
						on:click={reset}
					>
						{$i18n.t('Ask Again')}
					</button>
				</div>
			{/if}
		{:else}
			<!-- Browse by Use Case tab -->
			<p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
				{$i18n.t('Explore available models organized by what they do best.')}
			</p>

			<div class="space-y-1 max-h-[28rem] overflow-y-auto pr-1">
				{#each categorizedModels as { category, models: catModels }}
					<div class="rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
						<!-- Category header -->
						<button
							class="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-850 transition cursor-pointer"
							on:click={() => toggleCategory(category.key)}
						>
							<div class="text-gray-500 dark:text-gray-400">
								{#if expandedCategories[category.key]}
									<ChevronDown className="size-4" />
								{:else}
									<ChevronRight className="size-4" />
								{/if}
							</div>
							<div class="text-gray-600 dark:text-gray-300">
								<svelte:component this={category.icon} className="size-5" strokeWidth="1.5" />
							</div>
							<div class="flex-1 text-left">
								<div class="text-sm font-medium">{$i18n.t(category.label)}</div>
								<div class="text-xs text-gray-400 dark:text-gray-500">
									{$i18n.t(category.description)}
								</div>
							</div>
							<div
								class="text-xs text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full"
							>
								{catModels.length}
							</div>
						</button>

						<!-- Expanded model list -->
						{#if expandedCategories[category.key]}
							<div class="border-t border-gray-100 dark:border-gray-800">
								{#each catModels as m, i}
									<button
										class="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-gray-50 dark:hover:bg-gray-850 transition cursor-pointer
											{i < catModels.length - 1 ? 'border-b border-gray-50 dark:border-gray-800/50' : ''}"
										on:click={() => selectModel(m.id)}
									>
										<!-- Tree connector line -->
										<div class="w-4 flex items-center justify-center text-gray-300 dark:text-gray-600">
											{#if i < catModels.length - 1}
												<svg class="size-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
													<path d="M8 0 L8 16 M8 8 L16 8" />
												</svg>
											{:else}
												<svg class="size-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
													<path d="M8 0 L8 8 L16 8" />
												</svg>
											{/if}
										</div>
										<img
											src={`${WEBUI_API_BASE_URL}/models/model/profile/image?id=${encodeURIComponent(m.id)}&lang=${$i18n.language}`}
											alt={m.name ?? m.id}
											class="rounded-full size-6 flex-shrink-0 object-cover"
											loading="lazy"
										/>
										<div class="flex-1 text-left min-w-0">
											<div class="text-sm font-medium truncate">{m.name ?? m.id}</div>
											{#if m?.info?.meta?.description}
												<div class="text-xs text-gray-400 dark:text-gray-500 truncate">
													{m.info.meta.description}
												</div>
											{/if}
										</div>
									</button>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>
</Modal>
