import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    ParallelAgentRunner,
    ReActAgent,
)
from llama_index.core.query_engine import MultiStepQueryEngine, RetrieverQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RAGAgent:
    def __init__(self, cv_file_path, job_desc):
        self.llm = OpenAI(model="gpt-4o-mini")
        Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        self.job_desc = job_desc

        # load documents
        documents = SimpleDirectoryReader(input_files=[cv_file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        self.cv_nodes = splitter.get_nodes_from_documents(documents)

        documents = [Document(text=job_desc)]
        splitter = SentenceSplitter(chunk_size=1024)
        self.job_nodes = splitter.get_nodes_from_documents(documents)
        self.cv_vector_tool = None
        self.job_vector_tool = None
        self.cv_summary_tool = None
        self.job_summary_tool = None
        self._get_vector_tools()
        self._get_summary_tools()

    def _get_vector_tools(self):
        step_decompose_transform = StepDecomposeQueryTransform(
            llm=self.llm, verbose=True
        )

        if self.cv_vector_tool is None:
            cv_vector_index = VectorStoreIndex(self.cv_nodes)
            cv_vector_query_engine = RetrieverQueryEngine.from_args(
                retriever=cv_vector_index.as_retriever(), response_mode="compact"
            )
            self.cv_vector_tool = QueryEngineTool.from_defaults(
                name="cv_vector_tool",
                query_engine=cv_vector_query_engine,
                description=(
                    "Useful for retrieving specific context about user's CV/resume."
                ),
            )
            cv_vector_ms_query_engine = MultiStepQueryEngine(
                query_engine=cv_vector_query_engine,
                query_transform=step_decompose_transform,
                index_summary="Useful for retrieving specific context about user's CV/resume",
            )
            self.cv_vector_ms_tool = QueryEngineTool.from_defaults(
                name="cv_vector_ms_tool",
                query_engine=cv_vector_ms_query_engine,
                description=(
                    "Useful for retrieving specific context about user's CV/resume."
                ),
            )

        if self.job_vector_tool is None:
            job_vector_index = VectorStoreIndex(self.job_nodes)
            job_vector_query_engine = RetrieverQueryEngine.from_args(
                retriever=job_vector_index.as_retriever(), response_mode="compact"
            )
            self.job_vector_tool = QueryEngineTool.from_defaults(
                name="job_vector_tool",
                query_engine=job_vector_query_engine,
                description=(
                    "Useful for retrieving specific context about job posting user is applying to."
                ),
            )
            job_vector_ms_query_engine = MultiStepQueryEngine(
                query_engine=job_vector_query_engine,
                query_transform=step_decompose_transform,
                index_summary="Useful for retrieving specific context about job posting user is applying to.",
            )
            self.job_vector_ms_tool = QueryEngineTool.from_defaults(
                name="job_vector_ms_tool",
                query_engine=job_vector_ms_query_engine,
                description=(
                    "Useful for retrieving specific context about job posting user is applying to."
                ),
            )

    def _get_summary_tools(self):
        step_decompose_transform = StepDecomposeQueryTransform(
            llm=self.llm, verbose=True
        )

        if self.cv_summary_tool is None:
            cv_summary_index = SummaryIndex(self.cv_nodes)
            cv_summary_query_engine = cv_summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )
            self.cv_summary_tool = QueryEngineTool.from_defaults(
                name="cv_summary_tool",
                query_engine=cv_summary_query_engine,
                description=(
                    "Useful for summarization questions related to user's CV/resume."
                ),
            )
            cv_summary_ms_query_engine = MultiStepQueryEngine(
                query_engine=cv_summary_query_engine,
                query_transform=step_decompose_transform,
                index_summary="Useful for summarization questions related to user's CV/resume.",
            )
            self.cv_summary_ms_tool = QueryEngineTool.from_defaults(
                name="cv_summary_ms_tool",
                query_engine=cv_summary_ms_query_engine,
                description=(
                    "Useful for summarization questions related to user's CV/resume."
                ),
            )

        if self.job_summary_tool is None:
            job_summary_index = SummaryIndex(self.job_nodes)
            job_summary_query_engine = job_summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )
            self.job_summary_tool = QueryEngineTool.from_defaults(
                name="job_summary_tool",
                query_engine=job_summary_query_engine,
                description=(
                    "Useful for summarization questions related to job posting user is applying to."
                ),
            )
            job_summary_ms_query_engine = MultiStepQueryEngine(
                query_engine=job_summary_query_engine,
                query_transform=step_decompose_transform,
                index_summary="Useful for summarization questions related to job posting user is applying to.",
            )
            self.job_summary_ms_tool = QueryEngineTool.from_defaults(
                name="job_summary_ms_tool",
                query_engine=job_summary_ms_query_engine,
                description=(
                    "Useful for summarization questions related to job posting user is applying to."
                ),
            )

    def _get_response_from_react(self, agent, prompt):
        task = agent.create_task(prompt)
        step_output = agent.run_step(task.task_id)
        while not step_output.is_last:
            step_output = agent.run_step(task.task_id)
        response = agent.finalize_response(task.task_id)

        for step in task.extra_state["current_reasoning"]:
            print(step)

        thoughts = [
            step.thought
            for step in task.extra_state["current_reasoning"]
            if hasattr(step, "thought")
        ]
        prev_two_thoughts = "\n".join(thoughts[-2:])
        response = self.llm.complete(
            f"""
        Extract the tailored CV from the text below.
        Provide only the tailored CV and no other text:
                                    
        {prev_two_thoughts}
        {response}
        """
        )
        return str(response)

    def _auto_rag(self, prompt, use_react=False):
        if use_react:
            agent = ReActAgent.from_tools(
                [self.cv_vector_ms_tool, self.cv_summary_ms_tool],
                llm=self.llm,
                verbose=True,
                context=f"{self.job_desc}",
            )
            prompt = """Tailor the user's CV to be the best possible fit for the job description (provided as context). Do not invent information not in the user's CV.
                """
            return self._get_response_from_react(agent, prompt)
        else:
            agent_worker = FunctionCallingAgentWorker.from_tools(
                [
                    self.cv_vector_tool,
                    self.job_vector_tool,
                    self.cv_summary_tool,
                    self.job_summary_tool,
                ],
                llm=self.llm,
                verbose=True,
            )
            agent = ParallelAgentRunner(agent_worker)
            response = agent.query(prompt)
            return str(response)

    def _guided_rag(self, beginning_prompt, use_react=False):
        job_vector_query_engine = self.job_vector_tool.query_engine
        hard_skills = job_vector_query_engine.query(
            "What are the hard skills and tool knowledge required for this position?"
        )
        soft_skills = job_vector_query_engine.query(
            "What are the soft skills required for this position?"
        )
        quals = job_vector_query_engine.query(
            "What are the degrees required for this position?"
        )

        cv_vector_query_engine = self.cv_vector_tool.query_engine
        hard_skills_evid = cv_vector_query_engine.query(str(hard_skills))
        soft_skills_evid = cv_vector_query_engine.query(str(soft_skills))
        quals_evid = cv_vector_query_engine.query(str(quals))

        cvsum = self.cv_summary_tool.query_engine.query("cv summary")
        jobsum = self.job_summary_tool.query_engine.query("job summary")

        prompt = f"""{beginning_prompt}

            CV summary:
            {str(cvsum)}

            Job description summary:
            {str(jobsum)}

            Emphasize the following:
            {str(hard_skills_evid)}

            {str(soft_skills_evid)}

            {str(quals_evid)}"""

        if use_react:
            agent = ReActAgent.from_tools(
                [self.cv_vector_ms_tool, self.cv_summary_ms_tool],
                llm=self.llm,
                verbose=True,
                context=f"{self.job_desc}",
            )
            return self._get_response_from_react(agent, prompt)
        else:
            agent_worker = FunctionCallingAgentWorker.from_tools(
                [
                    self.cv_vector_tool,
                    self.job_vector_tool,
                    self.cv_summary_tool,
                    self.job_summary_tool,
                ],
                llm=self.llm,
                verbose=True,
            )
            agent = ParallelAgentRunner(agent_worker)
            response = agent.chat(prompt)
            return str(response)

    def rag_task(self, task_type: str):
        prompt = ""
        if "react" in task_type:
            cv_tools = "(user's CV summary provided in cv_summary_ms_tool and specific questions about user's CV answered using cv_vector_ms_tool)"
            job_tools = "(job description provided in context)"
        else:
            cv_tools = "(user's CV summary provided in cv_summary_tool and specific questions about user's CV answered using cv_vector_tool)"
            job_tools = "(job summary provided in job_summary_tool and specific questions about job answered using job_vector_tool)"
        if "cv" in task_type:
            prompt = f"Craft a new tailored CV using the user's CV {cv_tools} so that it is the best possible fit for the job description {job_tools}. Do not invent information not in the user's CV."
        elif "cover_letter" in task_type:
            prompt = f"Craft a cover letter using the user's CV {cv_tools} to demonstrate that the user is an ideal fit for the job {job_tools}. Do not invent information not in the user's CV."
        else:
            raise Exception("Invalid task type")

        if task_type.startswith("guided"):
            return self._guided_rag(prompt, "react" in task_type)
        elif task_type.startswith("auto"):
            return self._auto_rag(prompt, "react" in task_type)
        else:
            raise Exception("Invalid task type")
